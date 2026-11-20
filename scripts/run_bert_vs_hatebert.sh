#!/usr/bin/env bash
set -euo pipefail

TRAIN_CSV="${1:-data/train.csv}"
TEST_CSV="${2:-data/test.csv}"

python scripts/finetune_binary.py \
  --model bert-base-uncased \
  --train-csv "$TRAIN_CSV" \
  --test-csv "$TEST_CSV" \
  --output runs/bert-base

python scripts/finetune_binary.py \
  --model GroNLP/hateBERT \
  --train-csv "$TRAIN_CSV" \
  --test-csv "$TEST_CSV" \
  --output runs/hatebert
