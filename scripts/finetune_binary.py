#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    df = df.copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    values = set(df["label"].unique())
    if not values.issubset({0, 1}):
        raise ValueError(
            f"Expected binary labels 0/1, found: {sorted(values)}"
        )

    return df


def probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp[:, 1] / exp.sum(axis=1)


def evaluate(labels, probs, threshold: float) -> dict:
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels, preds, labels=[0, 1]
    ).ravel()

    out = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(
            f1_score(labels, preds, average="macro", zero_division=0)
        ),
        "harm_precision": float(
            precision_score(labels, preds, pos_label=1, zero_division=0)
        ),
        "harm_recall": float(
            recall_score(labels, preds, pos_label=1, zero_division=0)
        ),
        "harm_f1": float(
            f1_score(labels, preds, pos_label=1, zero_division=0)
        ),
        "no_harm_recall": float(
            recall_score(labels, preds, pos_label=0, zero_division=0)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if len(np.unique(labels)) == 2:
        out["roc_auc"] = float(roc_auc_score(labels, probs))
        out["pr_auc"] = float(average_precision_score(labels, probs))

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-csv", required=True, type=Path)
    ap.add_argument("--test-csv", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    train_df = load_csv(args.train_csv)
    test_df = load_csv(args.test_csv)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label={0: "NO_HARM", 1: "HARM"},
        label2id={"NO_HARM": 0, "HARM": 1},
    )

    def make_dataset(df: pd.DataFrame):
        ds = Dataset.from_pandas(
            df[["text", "label"]],
            preserve_index=False,
        )
        return ds.map(
            lambda batch: tokenizer(
                batch["text"],
                truncation=True,
                max_length=args.max_length,
                padding=False,
            ),
            batched=True,
        )

    train_ds = make_dataset(train_df)
    test_ds = make_dataset(test_df)

    training_args = TrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=10,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    trainer.train()

    pred = trainer.predict(test_ds)
    probs = probabilities_from_logits(pred.predictions)

    metrics = evaluate(
        test_df["label"].to_numpy(),
        probs,
        args.threshold,
    )

    predictions = test_df.copy()
    predictions["harm_probability"] = probs
    predictions["prediction"] = (
        predictions["harm_probability"] >= args.threshold
    ).astype(int)
    predictions["prediction_name"] = predictions["prediction"].map(
        {0: "no_harm", 1: "harm"}
    )

    predictions.to_csv(
        args.output / "test_predictions.csv",
        index=False,
    )

    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    trainer.save_model(args.output / "model")
    tokenizer.save_pretrained(args.output / "model")

    print(json.dumps(metrics, indent=2))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
