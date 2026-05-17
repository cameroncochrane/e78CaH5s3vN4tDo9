from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from binary_classification.config import MODELS_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

# Custom imports:
import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


@app.command()
def main(
    
    features_path: Path = PROCESSED_DATA_DIR / "feature_engineered_data", #X-train and X-test
    labels_path: Path = PROCESSED_DATA_DIR / "feature_engineered_data", #Y-train and Y-test (same directory as the X / features data)
    model_path: Path = MODELS_DIR / "model.pkl",
    
):
    
    def load_specific_pickle(directory, filename):
        filepath = Path(directory) / filename

        with open(filepath, "rb") as f:
            return pickle.load(f)
    
    X_train_scaled = load_specific_pickle(features_path, 'X_train_scaled.pkl')
    Y_train = load_specific_pickle(labels_path, "Y_train.pkl")

    xgb_param_grid = {
        "n_estimators": [200, 300, 500, 1000],
        "learning_rate": [0.001, 0.005,0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "subsample": [0.5, 0.6, 0.7, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0]
    }

    xgb_grid = GridSearchCV(
        estimator=XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=13
        ),
        param_grid=xgb_param_grid,
        cv=10,
        scoring="f1",
        n_jobs=-1
        )


    xgb_grid.fit(X_train_scaled, Y_train)


    print("Best Params:", xgb_grid.best_params_)
    print("Best CV F1:", xgb_grid.best_score_)

    best_xgb = xgb_grid.best_estimator_


    with open(model_path, 'wb') as f:
        pickle.dump(best_xgb, f)


if __name__ == "__main__":
    app()
