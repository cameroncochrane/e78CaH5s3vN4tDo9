from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import numpy as np

from binary_classification.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

# Custom imports:
import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    balanced_accuracy_score,
    log_loss
)


@app.command()
def main(

    features_path: Path = PROCESSED_DATA_DIR / "feature_engineered_data",
    labels_path: Path = PROCESSED_DATA_DIR / "feature_engineered_data",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "predictions", #Predictions has its own folder
    
):
    # Load model from the pkl file:
    with open(model_path, "rb") as f:
        xgb_model = pickle.load(f)
    
    # Load the specific pkl files (test datasets):
    def load_specific_pickle(directory, filename):
        filepath = Path(directory) / filename

        with open(filepath, "rb") as f:
            return pickle.load(f)
    

    X_test_scaled = load_specific_pickle(features_path, "X_test_scaled.pkl")
    Y_test = load_specific_pickle(labels_path, "Y_test.pkl")

    Y_xgb_grid_pred = xgb_model.predict(X_test_scaled)
    Y_xgb_grid_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

    def evaluate_binary_classifier(
            y_test,
            y_pred,
            y_proba=None,
            positive_label=1,
            verbose=True):
        """
        Evaluate a binary classification model.

        Parameters
        ----------
        y_test : array-like
            True labels.

        y_pred : array-like
            Predicted class labels.

        y_proba : array-like, optional
            Predicted probabilities for the positive class.
            Required for ROC-AUC and Log Loss.

        positive_label : int or str, default=1
            Label considered as the positive class.

        verbose : bool, default=True
            Whether to print results.

        Returns
        -------
        metrics_dict : dict
            Dictionary containing evaluation metrics.
        """

        # Convert to numpy arrays
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)

        # Core metrics
        metrics_dict = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
            "Precision": precision_score(
                y_test,
                y_pred,
                pos_label=positive_label,
                zero_division=0
            ),
            "Recall": recall_score(
                y_test,
                y_pred,
                pos_label=positive_label,
                zero_division=0
            ),
            "F1 Score": f1_score(
                y_test,
                y_pred,
                pos_label=positive_label,
                zero_division=0
            ),
            "Matthews Corrcoef": matthews_corrcoef(y_test, y_pred)
        }

        # Probability-based metrics
        if y_proba is not None:
            y_proba = np.array(y_proba)

            metrics_dict["ROC AUC"] = roc_auc_score(y_test, y_proba)
            metrics_dict["Log Loss"] = log_loss(y_test, y_proba)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Classification report
        clf_report = classification_report(y_test, y_pred)

        if verbose:
            print("\n===== Binary Classification Evaluation =====\n")

            for metric, value in metrics_dict.items():
                print(f"{metric}: {value:.4f}")

            print("\nConfusion Matrix:")
            print(cm)

            print("\nClassification Report:")
            print(clf_report)

        return {
            "metrics": metrics_dict,
            "confusion_matrix": cm,
            "classification_report": clf_report
    }

    xgb_eval = evaluate_binary_classifier(Y_test, Y_xgb_grid_pred, Y_xgb_grid_proba)

    predictions_path.mkdir(parents=True, exist_ok=True)
    
    # Save predictions, probabilities, and evaluation results
    with open(predictions_path / "Y_xgb_grid_pred.pkl", "wb") as f:
        pickle.dump(Y_xgb_grid_pred, f)
    
    with open(predictions_path / "Y_xgb_grid_proba.pkl", "wb") as f:
        pickle.dump(Y_xgb_grid_proba, f)
    
    with open(predictions_path / "xgb_eval.pkl", "wb") as f:
        pickle.dump(xgb_eval, f)
    

if __name__ == "__main__":
    app()
