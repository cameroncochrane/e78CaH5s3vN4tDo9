# Getting Started

This guide shows how to set up the project and run the full binary classification pipeline.

## Prerequisites

- Python 3.12
- A local clone of this repository
- The raw dataset at `data/raw/ACME-HappinessSurvey2020.csv`

## 1. Create and Activate a Virtual Environment

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

## 2. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Run the Pipeline

Run the scripts in this order from the repository root:

```bash
python binary_classification/dataset.py
python binary_classification/features.py
python binary_classification/modeling/train.py
python binary_classification/modeling/predict.py
python binary_classification/plots.py
```

## 4. Check Outputs

After running the pipeline, verify these artifacts:

- Processed data: `data/interim/ACME-HappinessSurvey2020_processed.csv`
- Feature-engineered train/test files:
	- `data/processed/feature_engineered_data/X_train_scaled.pkl`
	- `data/processed/feature_engineered_data/X_test_scaled.pkl`
	- `data/processed/feature_engineered_data/Y_train.pkl`
	- `data/processed/feature_engineered_data/Y_test.pkl`
- Trained model: `models/model.pkl`
- Predictions and evaluation:
	- `data/processed/predictions/Y_xgb_grid_pred.pkl`
	- `data/processed/predictions/Y_xgb_grid_proba.pkl`
	- `data/processed/predictions/xgb_eval.pkl`
- Figures: `reports/figures/`

## Optional Development Commands

Use the Makefile convenience targets:

```bash
make help
make requirements
make lint
make format
```

Note for Windows users: some targets use Unix-style commands (`find`) and may require Git Bash or WSL.
