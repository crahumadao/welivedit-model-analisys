# WeLivedIt Model Analysis

Small reproducible repository for analyzing the model choices used in WeLivedIt harm / no-harm classification.

This repository contains **scripts and aggregate evaluation results only**. Private WeLivedIt datasets, user-generated content, trained checkpoints, and model files are not included.

## Models

We compare three binary classifiers.

### BERT baseline

`bert-base-uncased`

BERT was fine-tuned on the original private WeLivedIt harm / no-harm training dataset.

### HateBERT baseline

`GroNLP/hateBERT`

HateBERT was fine-tuned using the **same training data and binary classification setup as the BERT baseline**.

This comparison tests whether starting from a language model already adapted to abusive online language provides an advantage over general-purpose BERT.

### WeLivedIt HateBERT

Our current model also starts from `GroNLP/hateBERT`, but is fine-tuned on a larger combined WeLivedIt training set containing the original training examples together with additional human-reviewed examples.

The combined training set contains approximately 802 examples.

No WeLivedIt training data is included in this repository.

## Internal evaluation

On the same private held-out test set:

| Model | Accuracy | Macro-F1 | Harm recall | Harm F1 | Non-harm recall |
|---|---:|---:|---:|---:|---:|
| BERT | 0.686 | 0.683 | 0.820 | 0.651 | 0.611 |
| HateBERT | 0.750 | 0.748 | 0.940 | 0.729 | 0.644 |
| WeLivedIt HateBERT | 0.771 | 0.768 | 0.920 | 0.742 | 0.689 |

These results suggest a modest improvement from BERT to HateBERT and a further small improvement from additional WeLivedIt fine-tuning.

Because the training datasets are relatively small and hyperparameter optimization has been limited, these results should be interpreted as preliminary rather than as a definitive model ranking.

## HateCheck evaluation

We additionally evaluate all three trained classifiers on the full HateCheck functional test suite.

HateCheck contains 3,728 test cases covering 29 hate-speech functionalities.

| Model | Accuracy | Macro-F1 | Harm recall | Non-harm accuracy | False-positive rate |
|---|---:|---:|---:|---:|---:|
| BERT | 0.687 | 0.469 | 0.965 | 0.075 | 0.925 |
| HateBERT | 0.701 | 0.463 | 0.995 | 0.056 | 0.944 |
| **WeLivedIt HateBERT** | **0.734** | **0.610** | 0.944 | **0.272** | **0.728** |

The vanilla HateBERT classifier has extremely high recall for hateful examples, but also produces many false positives on non-hateful examples.

Additional WeLivedIt fine-tuning substantially reduces this false-positive tendency while preserving high harmful-content recall.

The improvement is especially visible for several non-hateful phenomena such as reclaimed slurs, benign profanity, and identity-related statements.

Some performance is lost on obfuscated hateful expressions such as character swaps, added spaces, and leetspeak. This illustrates the trade-off introduced by domain-specific fine-tuning.

## Repository contents

```text
examples/
    sample_binary.csv

scripts/
    finetune_binary.py
    hatebert_cv.py
    run_bert_vs_hatebert.sh
    evaluate_hatecheck.py

results/
    hatecheck/
        bert_base/
        hatebert/
        welivedit_hatebert/
```

The files under `results/` contain aggregate metrics and per-functionality results only.

## Reproducibility

The model-training scripts are included to document the experimental setup.

The private WeLivedIt training data cannot be distributed, so the internal results cannot be reproduced exactly from this repository.

The HateCheck evaluation pipeline and public test suite can be reproduced independently.

## Interpretation

The current evidence does not imply that HateBERT or the WeLivedIt-adapted model is universally better than BERT.

Instead, the experiments suggest that:

1. HateBERT provides a useful domain-adapted initialization for harmful-content classification.
2. HateBERT alone tends to over-classify non-hateful examples as harmful.
3. Additional WeLivedIt fine-tuning improves the balance between harmful and non-harmful predictions.
4. This improvement comes with some degradation on particular obfuscated hateful-language phenomena.

The main objective is therefore not simply maximizing harmful-content recall, but finding a better balance between detecting harmful content and avoiding false positives.


## HateCheck summary

A compact comparison of the three evaluated models is available at:

`results/hatecheck/comparison.csv`

The main result is that HateBERT achieves very high harmful-content recall but also produces a very high false-positive rate on non-hateful cases.

The WeLivedIt-adapted HateBERT model reduces this false-positive tendency substantially while retaining high harmful-content recall, producing the best overall Macro-F1 of the three evaluated models.

This suggests that the additional WeLivedIt fine-tuning improves the balance between harmful and non-harmful classification rather than simply increasing sensitivity to harmful language.
