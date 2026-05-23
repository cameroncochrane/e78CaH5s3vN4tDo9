from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from binary_classification.config import MODELS_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

# Custom imports:
import pickle
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


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

    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
        "p": [1, 2]}

    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)

    knn_grid = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    refit=True
    )


    knn_grid.fit(X_train_scaled, Y_train)


    print("Best Params:", knn_grid.best_params_)
    print("Best CV F1:", knn_grid.best_score_)

    best_knn = knn_grid.best_estimator_


    with open(model_path, 'wb') as f:
        pickle.dump(best_knn, f)


if __name__ == "__main__":
    app()
