from pathlib import Path

#from loguru import logger
from tqdm import tqdm
import typer

from binary_classification.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

# Custom imports:
import pandas as pd


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "ACME-HappinessSurvey2020.csv",
    output_path: Path = PROCESSED_DATA_DIR / "ACME-HappinessSurvey2020_processed.csv",
    # ----------------------------------------------
):
    def generate_XY(data_path:str = input_path, feature_cols:list = None):

        # Import the raw data:
        raw_data = pd.read_csv(data_path)

        # Initial inspection of the data:
        print("Data Properties:")
        print(raw_data.info())

        print("Null Value Count:")
        print(raw_data.isnull().sum()) # Check for null/missing values
        print(raw_data.isnull().mean().sort_values(ascending=False)) # None present. Can proceed with EDA
   
        # Processing:
        # Define features and target columns:
        if feature_cols == None:
            feature_cols = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']
        
        target_cols = ['Y']

        raw_data.isnull().sum() # Remove any null values 
        raw_data.isnull().mean().sort_values(ascending=False)

        print("Basic Statistics:")
        print(raw_data.describe()) # Basic statistics
                

        X = raw_data[feature_cols]
        Y = raw_data[target_cols]

        data = pd.concat([X, Y], axis=1)
        print("Basic Statistics:")
        print(data.describe()) # Basic statistics

        return data, X, Y
    
    data, X, Y = generate_XY(input_path,None) #All feature columns will be exported to the 'processed' file

    data.to_csv(output_path, index=False)
    
    # -----------------------------------------

if __name__ == "__main__":
    app()
