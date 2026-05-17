# e78CaH5s3vN4tDo9

A binary classification project on the ACME Happy Customers dataset.

This repository contains an end-to-end workflow for:
- preparing raw survey data,
- performing feature selection and train/test preparation,
- training an XGBoost classifier,
- evaluating and saving predictions,
- generating exploratory and reporting plots.

## Quick Start

### 1. Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Run the pipeline

From the repository root:

```bash
python binary_classification/dataset.py
python binary_classification/features.py
python binary_classification/modeling/train.py
python binary_classification/modeling/predict.py
python binary_classification/plots.py
```

## Data and Outputs

### Input data
- Raw dataset expected at: `data/raw/ACME-HappinessSurvey2020.csv`

### Generated artifacts
- Processed dataset: `data/interim/ACME-HappinessSurvey2020_processed.csv`
- Feature-engineered train/test artifacts (pickle files):
    - `data/processed/feature_engineered_data/X_train_scaled.pkl`
    - `data/processed/feature_engineered_data/X_test_scaled.pkl`
    - `data/processed/feature_engineered_data/Y_train.pkl`
    - `data/processed/feature_engineered_data/Y_test.pkl`
- Trained model: `models/model.pkl`
- Predictions and evaluation objects:
    - `data/processed/predictions/Y_xgb_grid_pred.pkl`
    - `data/processed/predictions/Y_xgb_grid_proba.pkl`
    - `data/processed/predictions/xgb_eval.pkl`
- Figures saved to `reports/figures/` (for example: correlation matrix, boxplots, and distribution plots)

## Project Structure

```text
.
├── LICENSE
├── Makefile
├── README.md
├── pyproject.toml
├── requirements.txt
├── binary_classification/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── features.py
│   ├── plots.py
│   └── modeling/
│       ├── __init__.py
│       ├── predict.py
│       └── train.py
├── data/
│   ├── external/
│   ├── interim/
│   │   └── ACME-HappinessSurvey2020_processed.csv
│   ├── processed/
│   │   ├── feature_engineered_data/
│   │   └── predictions/
│   └── raw/
│       └── ACME-HappinessSurvey2020.csv
├── docs/
├── models/
├── notebooks/
├── references/
└── reports/
        └── figures/
```

## Script Overview

- `binary_classification/dataset.py`
    - Loads raw CSV, performs basic inspection, and writes an interim processed CSV.
- `binary_classification/features.py`
    - Runs feature diagnostics (correlation, variance threshold, mutual information, ANOVA),
        applies selected features, performs train/test split and robust scaling, and writes pickle artifacts.
- `binary_classification/modeling/train.py`
    - Trains an `XGBClassifier` with `GridSearchCV` and saves the best estimator.
- `binary_classification/modeling/predict.py`
    - Loads model and test data, computes predictions/probabilities, evaluates metrics,
        and saves prediction/evaluation pickle files.
- `binary_classification/plots.py`
    - Creates and saves visualizations to `reports/figures/`.

## Development Commands

The `Makefile` includes convenience targets:

```bash
make help
make requirements
make lint
make format
```

Note: some `Makefile` commands (for example `clean`) use Unix-style utilities (`find`), which may require Git Bash/WSL on Windows.

## Notebooks

Exploratory notebooks are available in `notebooks/`:
- `Happy_Customers_1.ipynb`
- `Happy_Customers_2.ipynb`
- `Happy_Customers_3.ipynb`
- `Happy_Customers_4.ipynb`

## Documentation

Project docs are managed with MkDocs in `docs/`.

```bash
mkdocs build
mkdocs serve
```

## License

This project is licensed under the terms in `LICENSE`.

