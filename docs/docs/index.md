# e78CaH5s3vN4tDo9 documentation

## Description

This project builds and evaluates a binary classifier for the ACME Happy Customers dataset.

## Workflow Summary

The project pipeline includes:

1. data processing from raw CSV,
2. feature engineering and train/test preparation,
3. model training with XGBoost + GridSearchCV,
4. prediction and evaluation,
5. plot generation for reporting.

For full setup instructions, see the Getting Started page.

## Common Commands

Run the full pipeline:

```bash
python binary_classification/dataset.py
python binary_classification/features.py
python binary_classification/modeling/train.py
python binary_classification/modeling/predict.py
python binary_classification/plots.py
```

Development helpers:

```bash
make help
make lint
make format
```

