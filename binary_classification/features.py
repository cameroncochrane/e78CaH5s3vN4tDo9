from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pickle

from binary_classification.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

# Custom imports:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

@app.command()
def main(
    
    input_path: Path = INTERIM_DATA_DIR / "ACME-HappinessSurvey2020_processed.csv",
    output_path: Path = PROCESSED_DATA_DIR / "feature_engineered_data"
    
):
    
    data = pd.read_csv(input_path) # Data has already gone through processing e.g. imputation etc.
    
    def split_scale_data(data, feature_cols, target_col, stratify, test_size = 0.15, random_state = 13):

        # Separate data into X and Y sets:

        X = data[feature_cols]
        Y = data[target_col]


        # Train-split:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, stratify=stratify, random_state=random_state)

        Y_train = Y_train["Y"].to_numpy().ravel()
        Y_test = Y_test["Y"].to_numpy().ravel()

        # Scale the data
        scaler = RobustScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)


        return X_train_scaled, X_test_scaled, Y_train, Y_test

    # Feature selection:
    feature_cols = data.columns.drop('Y')
    target_col = "Y"
    X = data[feature_cols]
    Y = data[target_col]

    #1. Correlation Analysis:
    print("Correlation Matrix")
    corr_matrix = data.corr(numeric_only=True)
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                linewidths=0.5
                )
    plt.title("Correlation Matrix")
    plt.show()


    # 2. Variance Theshold:
    from sklearn.feature_selection import VarianceThreshold
    vt_selector = VarianceThreshold(threshold=0.01)
    vt_selector.fit(X)
    selected_v_features = X.columns[vt_selector.get_support()]
    removed_v_features = X.columns[~vt_selector.get_support()]
    print("Selected:")
    print(selected_v_features)
    print("\nRemoved:")
    print(removed_v_features)

    # 3. Mutual Information
    from sklearn.feature_selection import mutual_info_classif
    Y_array = Y # Seems to complain when we pass a df column and not a 1d array for Y...
    mi_scores = mutual_info_classif(X, Y_array)
    print(mi_scores)
    # Put MI scores into a sorted table with relative contribution (%)
    mi_df = pd.DataFrame({
        "Feature": X.columns,
        "MI Score": mi_scores
    }).sort_values("MI Score", ascending=False).reset_index(drop=True)
    mi_total = mi_df["MI Score"].sum()
    mi_df["Relative Importance (%)"] = (
        (mi_df["MI Score"] / mi_total * 100) if mi_total > 0 else 0
    ).round(2)
    print(mi_df)


    # 4. ANOVA F-test:
    from sklearn.feature_selection import SelectKBest, f_classif
    # Fit selector
    af_selector = SelectKBest(score_func=f_classif, k="all")
    af_selector.fit(X, Y_array)
    # Create results dataframe
    anova_results = pd.DataFrame({
        "Feature": X.columns,
        "F_Score": af_selector.scores_,
        "P_Value": af_selector.pvalues_
    })
    # Sort by importance
    anova_results = anova_results.sort_values(
        by="F_Score",
        ascending=False
    )
    print(anova_results)


    # Update feature column list to consider the features to be used for modelling:
    feature_cols = ['X1','X2','X4','X6'] # Post feature selection columns (removed 'X3' and 'X5')
    target_col = ['Y'] # Also referred to as labels...

    # Processing the data (features selected) ready for modelling:
    X_train_scaled, X_test_scaled, Y_train, Y_test = split_scale_data(data=data,
                                                                      feature_cols=feature_cols,
                                                                      target_col=target_col,
                                                                      stratify=data[target_col],
                                                                      test_size=0.15,
                                                                      random_state=13)
    
    # Save the split and scaled data as pkl files to maintain data strcuture (rather than csv).
    # Add to the 'data/processed/feature_engineered_data' folder to be accessed by other scripts for model training and testing/evaluation
    train_test_files = {
    "X_train_scaled.pkl": X_train_scaled,
    "X_test_scaled.pkl": X_test_scaled,
    "Y_train.pkl": Y_train,
    "Y_test.pkl": Y_test
    }
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    for filename, data_obj in train_test_files.items():
        filepath = output_path / filename
        with open(filepath, 'wb') as f:
            pickle.dump(data_obj, f)

    


if __name__ == "__main__":
    app()
