#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def softmax_harm(logits):
    logits = np.asarray(logits)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp[:, 1] / exp.sum(axis=1)


def metrics(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    non_harm_total = tn + fp
    harm_total = tp + fn

    return {
        "n": int(len(y)),
        "accuracy": float(
            accuracy_score(y, pred)
        ),
        "macro_f1": float(
            f1_score(
                y,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        "harm_precision": float(
            precision_score(
                y,
                pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "harm_recall": float(
            recall_score(
                y,
                pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "harm_f1": float(
            f1_score(
                y,
                pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "non_harm_accuracy": (
            float(tn / non_harm_total)
            if non_harm_total
            else None
        ),
        "harm_accuracy": (
            float(tp / harm_total)
            if harm_total
            else None
        ),
        "false_positive_rate": (
            float(fp / non_harm_total)
            if non_harm_total
            else None
        ),
        "false_negative_rate": (
            float(fn / harm_total)
            if harm_total
            else None
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        required=True,
    )

    ap.add_argument(
        "--name",
        required=True,
    )

    ap.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "data/public/hatecheck_test_suite.csv"
        ),
    )

    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/hatecheck"),
    )

    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    ap.add_argument(
        "--max-length",
        type=int,
        default=128,
    )

    ap.add_argument(
        "--quick",
        action="store_true",
    )

    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    required = {
        "functionality",
        "test_case",
        "label_gold",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing HateCheck columns: "
            f"{sorted(missing)}"
        )

    df = df.copy()

    df["label"] = (
        df["label_gold"]
        .astype(str)
        .str.lower()
        .map({
            "non-hateful": 0,
            "hateful": 1,
        })
    )

    if df["label"].isna().any():
        unknown = (
            df.loc[
                df["label"].isna(),
                "label_gold",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unknown HateCheck labels: {unknown}"
        )

    df["label"] = df["label"].astype(int)

    if args.quick:
        # Deterministic small diagnostic:
        # up to 10 examples per functionality.
        rng = np.random.RandomState(42)

        df = (
            df.assign(
                _quick_random=rng.random(len(df))
            )
            .sort_values(
                ["functionality", "_quick_random"]
            )
            .groupby(
                "functionality",
                group_keys=False,
            )
            .head(10)
            .drop(columns=["_quick_random"])
            .reset_index(drop=True)
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("device:", device)
    print("model:", args.model)
    print("cases:", len(df))
    print(
        "functionalities:",
        df["functionality"].nunique(),
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            args.model,
            use_fast=True,
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(args.model)
        .to(device)
    )

    model.eval()

    probs = []

    texts = (
        df["test_case"]
        .astype(str)
        .tolist()
    )

    for start in range(
        0,
        len(texts),
        args.batch_size,
    ):
        batch_texts = texts[
            start:
            start + args.batch_size
        ]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )

        encoded = {
            k: v.to(device)
            for k, v in encoded.items()
        }

        with torch.no_grad():
            output = model(**encoded)

        batch_probs = softmax_harm(
            output.logits
            .detach()
            .cpu()
            .numpy()
        )

        probs.extend(
            batch_probs.tolist()
        )

    df["harm_probability"] = probs

    df["prediction"] = (
        df["harm_probability"]
        >= args.threshold
    ).astype(int)

    df["correct"] = (
        df["prediction"]
        == df["label"]
    )

    overall = metrics(
        df["label"],
        df["prediction"],
    )

    overall.update({
        "model_name": args.name,
        "model_path": args.model,
        "threshold": args.threshold,
        "quick": args.quick,
        "functionalities":
            int(
                df["functionality"]
                .nunique()
            ),
    })

    by_functionality = []

    for functionality, part in (
        df.groupby("functionality")
    ):
        m = metrics(
            part["label"],
            part["prediction"],
        )

        m.update({
            "functionality":
                functionality,

            "gold_label":
                part["label_gold"]
                .iloc[0],

            "model":
                args.name,
        })

        by_functionality.append(m)

    by_functionality_df = (
        pd.DataFrame(by_functionality)
        .sort_values(
            [
                "accuracy",
                "functionality",
            ]
        )
    )

    output_dir = (
        args.output_root
        / args.name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_dir
        / "metrics.json"
    ).write_text(
        json.dumps(
            overall,
            indent=2,
        ),
        encoding="utf-8",
    )

    by_functionality_df.to_csv(
        output_dir
        / "by_functionality.csv",
        index=False,
    )

    # Useful locally for error analysis.
    # We do NOT need to publish this file.
    df.to_csv(
        output_dir
        / "predictions.csv",
        index=False,
    )

    print()
    print("=== OVERALL ===")
    print(
        json.dumps(
            overall,
            indent=2,
        )
    )

    print()
    print(
        "=== WORST HATEFUL FUNCTIONALITIES ==="
    )

    print(
        by_functionality_df[
            by_functionality_df["gold_label"] == "hateful"
        ][
            [
                "functionality",
                "n",
                "accuracy",
                "false_negative_rate",
            ]
        ]
        .sort_values(
            ["accuracy", "functionality"]
        )
        .head(15)
        .to_string(index=False)
    )

    print()
    print(
        "=== WORST NON-HATEFUL FUNCTIONALITIES ==="
    )

    print(
        by_functionality_df[
            by_functionality_df["gold_label"] == "non-hateful"
        ][
            [
                "functionality",
                "n",
                "accuracy",
                "false_positive_rate",
            ]
        ]
        .sort_values(
            ["accuracy", "functionality"]
        )
        .head(15)
        .to_string(index=False)
    )

    print()
    print("saved:", output_dir)


if __name__ == "__main__":
    main()
