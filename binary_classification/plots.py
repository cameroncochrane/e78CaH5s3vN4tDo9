from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from binary_classification.config import FIGURES_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

# Custom imports:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@app.command()
def main(
    
    input_path: Path = INTERIM_DATA_DIR / "ACME-HappinessSurvey2020_processed.csv", # Use the processed data pre-feature engineering (not the feature engineered set(s))
    output_path: Path = FIGURES_DIR # Need individual paths for each plot
    
):
    data = pd.read_csv(input_path)
    feature_cols = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']
    target_cols = ['Y']
    X = data[feature_cols]
    Y = data[target_cols]
    
    # Value Distribution (Gridplot)
    data.hist(figsize=(12, 8), bins=5, edgecolor='black')
    plt.suptitle("Value Distribution (Gridplot)")
    plt.savefig(output_path / "value_distribution_gridplot.png", dpi=300, bbox_inches='tight')


    # Target Class (Y) vs Individual Feature (X1-X6) (Gridplot)
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    axes = axes.ravel()
    for idx, col in enumerate(X.columns):
        axes[idx].scatter(X[col], Y['Y'], alpha=0.6)
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Y')
        axes[idx].set_title(f'Y vs {col}')
        axes[idx].grid(True, alpha=0.3)
    fig.suptitle('Target Class (Y) vs Individual Feature (X1-X6)', fontsize=14)
    plt.savefig(output_path / "target_vs_feature_scatter_gridplot.png", dpi=300, bbox_inches='tight')


    # Distribution of Target Class (Y) by Feature (X1-X6) Value (Gridplot)
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    axes = axes.ravel()
    for idx, col in enumerate(X.columns):
        df_plot = pd.DataFrame({col: X[col], 'Y': Y['Y']})
        df_plot.groupby([col, 'Y']).size().unstack(fill_value=0).plot(kind='bar', ax=axes[idx])
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Count')
        axes[idx].set_title(f'Y vs {col}')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].tick_params(axis='x', rotation=0)
    fig.suptitle('Distribution of Target Class (Y) by Feature (X1-X6) Value', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / "target_vs_feature_distribution_gridplot.png", dpi=300, bbox_inches='tight')


    # Data Correlation (Gridplot)
    g = sns.pairplot(data, hue='Y', diag_kind='kde')
    g.figure.set_size_inches(12, 8)
    g.figure.suptitle("Data Correlation (Gridplot)", y=0.99)
    g.figure.subplots_adjust(top=0.95)
    plt.savefig(output_path / "data_correlation_gridplot.png", dpi=300, bbox_inches='tight')


    # Outlier Detection by Y value
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    axes = axes.flatten()
    for idx, feature in enumerate(feature_cols):
        sns.boxplot(x='Y', y=feature, data=data, ax=axes[idx])
        axes[idx].set_xlabel('Y')
        axes[idx].set_ylabel(feature)
    fig.suptitle("Outlier Detection by Y value", y=1.00)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path / "boxplots_outlier_detection_by_Y.png", dpi=300, bbox_inches='tight')

    # Correlation Matrix (feature selection)
    corr_matrix = data.corr(numeric_only=True)
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                linewidths=0.5
                )
    plt.title("Correlation Matrix")
    plt.savefig(output_path / "correlation_matrix.png", dpi=300, bbox_inches='tight')

    

if __name__ == "__main__":
    app()