
# Winter Wheat Yield Prediction App

## Overview

This Streamlit application predicts winter wheat yield in Ukraine using machine learning. It integrates weather data, NDVI satellite indices, and field operations (e.g., fertilizers, works). The app supports task description, exploratory data analysis (EDA), model training, validation with metrics and SHAP explanations, and forecasting on new data.

Key features:

-   **Data Preprocessing**: Cleaning, embedding textual operations, spatial and semantic clustering (UMAP + HDBSCAN).
-   **Modeling**: LightGBM for mean predictions and quantile regression (5% lower, 95% upper bounds for 90% CI).
-   **Weekly Training**: Models trained from week 20 to 28, excluding future features.
-   **Interpretability**: SHAP values for feature importance.
-   **Forecasting**: Upload new data for predictions with interactive filters and visualizations.

## Installation

1.  Clone the repository:
    ```
    git clone https://github.com/herbvertuher/Wheat-Yield-Prediction-Kernel-UCU.git
    cd Wheat-Yield-Prediction-Kernel-UCU
    ```
2.  Install dependencies:
    ```
    conda env create -f conda_env.yml
    ```
3.  Run the app:    
    ```
    conda activate yield-ucu & streamlit run app.py
    ```
    **OR**
    
    Use `streamlit_run.bat` file.
    

## Usage

-   **Tabs Overview**:
    -   **Task Description**: Project goals, data details, and expected outcomes.
    -   **Exploratory Data Analysis**: Preprocessing steps, spatial clustering (dendrogram, agglomerative results), operation embeddings (UMAP/HDBSCAN/t-SNE visualizations).
    -   **Training**: Model pipeline, hyperparameters, weekly CV metrics (MAE, MAPE, R2, Pinball loss) with plots.
    -   **Validation**: Holdout metrics, SHAP summaries, confidence interval visualizations.
    -   **Forecasting**: Upload main/operations Parquet files, compute predictions, view metrics table, filterable predictions DataFrame, and interactive CI plots.
-   **Forecasting Example**:
    1.  Upload files.
    2.  View computed metrics.
    3.  Filter predictions by week/field.
    4.  Visualize selected forecasts.

## Data

-   Training data: df_2025_v2_extended_weather.parquet (weather/NDVI) and operations_2025.parquet (field activities).
-   Artifacts (models, reducers): Stored in /artifacts/ for loading.

## Dependencies

-   Python 3.11+
-   See conda_env.yml for full list.

## License

MIT License. See LICENSE for details.

## Contact

For questions, open an issue or contact [[yehor@duck.com](mailto:yehor@duck.com)].