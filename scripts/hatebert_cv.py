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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


MODEL = "GroNLP/hateBERT"


def probs_from_logits(logits):
    logits = np.asarray(logits)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp[:, 1] / exp.sum(axis=1)


def evaluate(y, probs, threshold):
    y = np.asarray(y, dtype=int)
    probs = np.asarray(probs, dtype=float)
    pred = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    result = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(
            f1_score(y, pred, average="macro", zero_division=0)
        ),
        "harm_precision": float(
            precision_score(y, pred, pos_label=1, zero_division=0)
        ),
        "harm_recall": float(
            recall_score(y, pred, pos_label=1, zero_division=0)
        ),
        "harm_f1": float(
            f1_score(y, pred, pos_label=1, zero_division=0)
        ),
        "no_harm_recall": float(
            recall_score(y, pred, pos_label=0, zero_division=0)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if len(np.unique(y)) == 2:
        result["roc_auc"] = float(roc_auc_score(y, probs))
        result["pr_auc"] = float(
            average_precision_score(y, probs)
        )

    return result


def best_threshold(y, probs):
    best = None

    for threshold in np.arange(0.10, 0.91, 0.01):
        metrics = evaluate(y, probs, threshold)

        if best is None or metrics["f1_macro"] > best["f1_macro"]:
            best = metrics

    return best


def tokenize_dataframe(df, tokenizer, max_length):
    ds = Dataset.from_pandas(
        df[["text", "label"]],
        preserve_index=False,
    )

    return ds.map(
        lambda batch: tokenizer(
            batch["text"].astype(str)
            if hasattr(batch["text"], "astype")
            else batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        ),
        batched=True,
    )


def run_fold(
    train_df,
    val_df,
    learning_rate,
    output_dir,
    seed,
    epochs,
    batch_size,
    max_length,
):
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        use_fast=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        num_labels=2,
        id2label={0: "NO_HARM", 1: "HARM"},
        label2id={"NO_HARM": 0, "HARM": 1},
    )

    train_ds = tokenize_dataframe(
        train_df,
        tokenizer,
        max_length,
    )

    val_ds = tokenize_dataframe(
        val_df,
        tokenizer,
        max_length,
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=10,
        logging_steps=20,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        seed=seed,
        data_seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    trainer.train()

    pred = trainer.predict(val_ds)
    return probs_from_logits(pred.predictions)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", default="GroNLP/hateBERT")
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--learning-rate", required=True, type=float)

    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--grouped", action="store_true")
    ap.add_argument("--output-root", type=Path, default=Path("runs/hatebert_cv"))

    args = ap.parse_args()

    global MODEL
    MODEL = args.model

    df = pd.read_csv(args.csv)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    run_dir = (
        args.output_root
        / args.name
        / f"lr_{args.learning_rate:g}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    y = df["label"].to_numpy()

    if args.grouped:
        if "post_group" not in df.columns:
            raise ValueError(
                "--grouped requires a post_group column"
            )

        splitter = StratifiedGroupKFold(
            n_splits=args.folds,
            shuffle=True,
            random_state=args.seed,
        )

        splits = splitter.split(
            df,
            y,
            groups=df["post_group"],
        )

    else:
        splitter = StratifiedKFold(
            n_splits=args.folds,
            shuffle=True,
            random_state=args.seed,
        )

        splits = splitter.split(df, y)

    oof_probs = np.zeros(len(df), dtype=float)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        print()
        print("=" * 70)
        print(
            f"{args.name} | LR={args.learning_rate} | fold {fold}/{args.folds}"
        )
        print("=" * 70)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        probs = run_fold(
            train_df=train_df,
            val_df=val_df,
            learning_rate=args.learning_rate,
            output_dir=run_dir / f"fold_{fold}",
            seed=args.seed + fold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        oof_probs[val_idx] = probs

        fm = evaluate(
            val_df["label"].to_numpy(),
            probs,
            0.5,
        )
        fm["fold"] = fold
        fold_metrics.append(fm)

        print(json.dumps(fm, indent=2))

    threshold_metrics = best_threshold(
        y,
        oof_probs,
    )

    oof_df = df.copy()
    oof_df["harm_probability"] = oof_probs
    oof_df["prediction"] = (
        oof_probs >= threshold_metrics["threshold"]
    ).astype(int)

    oof_df.to_csv(
        run_dir / "oof_predictions.csv",
        index=False,
    )

    summary = {
        "model": MODEL,
        "dataset": args.name,
        "n": len(df),
        "learning_rate": args.learning_rate,
        "folds": args.folds,
        "epochs": args.epochs,
        "grouped": args.grouped,
        "fold_metrics_at_0.5": fold_metrics,
        "oof_best_threshold_metrics": threshold_metrics,
    }

    (run_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("FINAL OOF")
    print(json.dumps(threshold_metrics, indent=2))
    print("saved:", run_dir)


if __name__ == "__main__":
    main()
