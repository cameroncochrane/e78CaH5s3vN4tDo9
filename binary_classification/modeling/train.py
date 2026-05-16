from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from binary_classification.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "feature_engineered_data", #X-train and X-test
    labels_path: Path = PROCESSED_DATA_DIR / "feature_engineered_data", #Y-train and Y-test (same directory as the X / features data)
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    
    # -----------------------------------------


if __name__ == "__main__":
    app()
