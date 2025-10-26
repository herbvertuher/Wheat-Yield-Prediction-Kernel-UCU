from pathlib import Path
import streamlit as st
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# st.set_page_config(layout="wide")

root_path = Path(__file__).resolve().parents[0]

artifacts_path = root_path / 'artifacts'
plots_path = root_path / 'plots'
metrics_path = root_path / 'metrics'
confidence_intervals_path = plots_path / 'confidence_intervals'
shap_path = plots_path / 'shap'

train_coinfig = {}
with open(artifacts_path / 'train_config.json', 'r', encoding='utf-8') as f:
    train_coinfig = json.load(f)

image_paths = {
    'Distribution_number_of_operations': plots_path / 'Distribution_number_of_operations.png',
    'Dendrogram': plots_path / 'Dendrogram_visualization.png',
    'AgglomerativeClustering': root_path / 'plots' / 'AgglomerativeClustering_results_visualization.png',
    'UMAP_HDBSCAN_2D': plots_path / 'umap_hdbscan_2D_visualization.png',
    'UMAP_HDBSCAN_2D_Alibaba': plots_path / 'umap_hdbscan_2D_visualization_Alibaba.png',
    't-SNE': plots_path / 't-SNE_visualization.png',
    't-SNE_Alibaba': plots_path / 't-SNE_visualization_Alibaba.png',
    'Model_Mean_accuracy_CV': plots_path / 'Model_Mean_accuracy_CV.png',
    'Models_Lower_and_Upper_accuracy_CV': plots_path / 'Models_Lower_and_Upper_accuracy_CV.png',
}

df_cv_metrics = pd.read_pickle(metrics_path / 'df_cv_metrics.pkl')
df_val_metrics = pd.read_pickle(metrics_path / 'df_val_metrics.pkl')



# Створення вкладок (tabs) для організації
tabs = st.tabs(['Task Description', 'Exploratory Data Analysis', 'Training', 'Validation', 'Forecasting'])
tab_idx = 0

with tabs[tab_idx]:  # Або назвіть відповідну вкладку

    st.markdown("""
    ### Background
    `Yield prediction` stands as a critical challenge in agricultural, particularly for winter wheat, where early and accurate forecasts are essential. These predictions allow for improved planning in agribusiness operations, logistics, and trading activities. By combining weather observations, satellite indices like NDVI, and farm-level operational data, modern methods enhance accuracy and provide valuable insights into the factors influencing productivity. `Solving this problem` not only boosts efficiency in resource allocation and decision-making but also leads to optimized harvests, reduced risks, and better economic outcomes for stakeholders.

    In this project, I was provided with `two datasets` to support the analysis: `one` containing weekly cultivation data, including weather metrics and NDVI indices, and `another` detailing the operations conducted on the fields, such as fertilizer applications and field works. Together, these datasets encompassed `783 rows`, each representing a `distinct agricultural field` in Ukraine.

    **Project Goals**  
    The `primary objectives` include developing a `baseline model` to predict winter wheat yield using weather data and satellite-based NDVI indices. This is followed by an `extended model` that incorporates operational data like fertilizer applications and field works, allowing for a comparison of performance improvements against the baseline. Additionally, a `bonus task` involves constructing a `confidence interval` for the yield predictions.

    **Data Provided**  
    The `target variable` is winter wheat `yield` measured in tons per hectare (t/ha). `Key features` span several categories: weather data includes temperature, precipitation, evapotranspiration, wind speed, and sunshine duration; remote sensing provides weekly NDVI indices from Sentinel-2 satellites; spatial information consists of field centroid coordinates (latitude and longitude); and `operations` cover fertilizer applications (type, amount, and date) along with field works (name, date, and covered area).

    **Expected Outcomes**  
    - Train yield prediction models weekly, starting from two months before harvest up to the harvest date, to assess how predictive accuracy evolves over time.  
    - Compare the performance of models with and without operational data to highlight the value added by including farm-level activities.  
    - Use `SHAP` (SHapley Additive exPlanations) to identify the most influential features in the predictions.  
    - Construct a `CI` (confidence interval) for the total harvest volume across all fields.

    **Usage and Impact**  
    The insights from this project will benefit various departments. For agribusiness, it means enhanced field management decisions; logistics teams can better plan harvest operations and allocate resources; and trading departments will gain a stronger basis for decisions tied to expected production volumes.
    """)
    tab_idx += 1

with tabs[tab_idx]:

    # === Data Loading and Initial Overview ===
    st.markdown("""
    ### Data Loading and Initial Overview
    In this section, I describe the preprocessing steps I applied to the provided datasets to prepare them for yield prediction modeling. I was provided with two main datasets: the primary dataset `df_main` from `'data/df_2025_v2_extended_weather.parquet'`, containing weekly data such as weather metrics and NDVI indices, and the operations dataset `df_operations` from `'data/operations_2025.parquet'`, detailing field activities like fertilizer applications and works. Together, these encompass data for `783 agricultural fields` in Ukraine.
    """)
    
    # === Spatial Clustering for Stratified Data Splitting ===
    st.markdown("""
    ### Spatial Clustering for Stratified Data Splitting
                
    To ensure more robust data splitting and cross-validation, given the spatial nature of agricultural fields where nearby fields share similar characteristics, I introduced a `'Region_cluster'` feature based on geographic coordinates. This helps stabilize splits by stratifying on regional groups, reducing variance in performance estimates.

    First, I transformed the `Latitude` and `Longitude` from `EPSG:4326 (WGS84)` to `EPSG:32636 (UTM zone for Ukraine)` using `pyproj.Transformer` for accurate Cartesian coordinates. I then scaled them by dividing by 1,000 to work in kilometers and standardized with `StandardScaler` to prepare for clustering.

    I performed `hierarchical clustering` with the `'ward'` linkage method, which minimizes intra-cluster variance during merges—the most commonly recommended approach. The linkage matrix was computed on scaled coordinates, and I visualized the results with a `dendrogram`, showing `Euclidean distances` without leaf labels for clarity.
    """)
    
    st.image(image_paths['Dendrogram'], caption=f"Dendrogram Plot", width='stretch')

    st.markdown("""
    Based on the dendrogram, I selected `4 clusters` and applied `AgglomerativeClustering` with `'ward'` linkage and `Euclidean` metric, assigning labels to region_labels. I plotted the clustered points in scaled space. As an alternative, I considered `KMeans` with 4 clusters but opted for the hierarchical method for its interpretability.
    """)
    
    st.image(image_paths['AgglomerativeClustering'], caption=f"Agglomerative Clustering Results Plot", width='stretch')

    st.markdown("""
    I added `'Region_cluster'` as a categorical column. This feature was used for stratified splitting: in `train_test_split` with test_size=0.15 and stratify=X['Region_cluster'], and in `StratifiedKFold` with `n_splits` folds, ensuring balanced regional representation across folds for more reliable and stable model evaluations.
    """)

    # === Preprocessing `df_main` ===
    st.markdown("""
    ### Preprocessing `df_main`
    I performed several cleaning steps on the `df_main` to focus on relevant features for winter wheat yield prediction. First, I filtered out rows with missing `Culture` values to ensure data relevance. I dropped columns related to the previous year, identified by the `'_prev_year'` suffix, as the analysis targets the current season.

    Next, I removed columns with numeric suffixes greater than or equal to 34, extracted via regular expression on column names, to exclude from the features weeks in which the harvest has already taken place. I excluded rows lacking `'Area'` information and dropped specific non-essential columns such as `'Year'`, `'Culture'`, `'Moisture'`, and `'crop'` to streamline the dataset.

    Weather features for `25 weeks` (from week 9 to week 33):

    1. ndvi_mean
    2. CumSum_Precipitation
    3. CumSum_TempEffective
    4. EvapoTranspiration_avg
    5. PrecipitationCumulative_sum
    6. Sunshine_duration_avg
    7. TempEffective
    8. TempStandardMax
    9. TempStandardMin
    10. Temp_avg
    11. WindSpeed
    
    General field features:
                
    1. Area
    2. N
    3. P
    4. K
    5. Latitude
    6. Longitude
            
    Counting the total number of features: `(11 × 25) + 6 = 281`. Resulting dimension of the dataset: `rows: 783`, `columns: 281`

    Additionally, I eliminated 6 weather features to reduce redundancy and noise: these included `'CumSum_Precipitation'`, `'CumSum_TempEffective'`, `'WindSpeed'`, `'Sunshine_duration_avg'`, `'TempStandardMin'`, and `'TempStandardMax'`. The removed features had low significance according to the SHAP values, and their removal allowed to improve the predictive ability of the model and reduce the dimensionality. This resulted in a cleaner dataset focused on key weather, NDVI, and spatial features like temperature, precipitation, evapotranspiration, and field coordinates.
    
    Removing these features allowed me to reduce the feature size by: 6 × 25 = `150`.
    """)

    # === Preprocessing `df_operations` ===
    st.markdown("""
    ### Preprocessing `df_operations`
    For `df_operations`, I renamed columns for clarity: to `'operation'`, `'detailed_operation'`, `'year_of_start'`, `'date_of_start'`, `'total'`, and `'field_id'`. I converted the `'date_of_start'` column to datetime format for temporal handling.

    I primarily used the `'detailed_operation'` column (as set in `operation_column`) for detailed categorization, though `'operation'` could serve as an alternative. I cleaned operations names by removing periods for consistency.

    ### Merging Close-Dated Operations
    To handle potential duplicated or closely timed entries, I grouped operations by `'field_id'`, `'operation'`, and `'detailed_operation'`. Within each group, I sorted rows by `'date_of_start'`, and merged entries within a `gap_in_days` of 30 by summing their `'total'` values, using the earliest date as the representative. This aggregation, implemented via a custom function `merge_close_dates`, produced `df_operations_merged`, reducing noise from minor date variations and consolidating similar activities.

    Finally, I created a summarized DataFrame by grouping merged operations by `'field_id'` and concatenating the `'detailed_operation'` values with periods for a compact per-field operation history. This prepared the operations data for integration into the extended model, enabling analysis of their impact on yield alongside weather and NDVI features.
    
    ### Exploring Feature Representations for Operations
    After initial preprocessing of the `df_operations` DataFrame, I experimented with various methods to transform the textual operation data into numerical features suitable for integration into the yield prediction models. I started with simpler bag-of-words techniques and progressed to more advanced embedding-based approaches, aiming to capture the semantic richness of the operations while reducing dimensionality and identifying patterns.
    """)
    
    st.image(image_paths['Distribution_number_of_operations'], caption=f"Distribution of the number of operations per field", width='stretch')     
    
    st.markdown("""   
    #### Simpler Approaches: Bag of Words and TF-IDF
    I began with a `Bag of Words` (BoW) representation using `CountVectorizer` from scikit-learn. To handle the Ukrainian text effectively, I implemented a custom preprocessor to remove parentheses and periods, and a tokenizer that captured patterns like numerical ranges (e.g., `'5-8 см'`), ordinal indicators (e.g., `'1-ша'`), abbreviations (e.g., `'б/трави'`), and standard words. I also filtered out common stopwords such as `'на'`, `'та'`, `'із'`, and `'з'`. 

    Next, I applied `Term Frequency-Inverse Document Frequency` (TF-IDF) using `TfidfVectorizer` from scikit-learn. A custom `clean_text` function was implemented to preprocess the input by converting to lowercase, stripping leading/trailing whitespace, removing punctuation (such as parentheses, quotes, and commas), normalizing multiple spaces to single ones, and standardizing dash variants. Tokenization was handled via a bespoke `uk_tokenizer`, utilizing regular expressions to identify sequences of Ukrainian letters, digits, and hyphens.

    `Result:` These methods provided straightforward representations but risked high dimensionality and loss of semantic context. Furthermore, the features derived from operations encoded in this manner did not rank among the top important features according to SHAP values, indicating limited impact on model predictions. This prompted me to explore more sophisticated techniques.
    
    #### Advanced Approaches: Embeddings with Dimensionality Reduction and Clustering
    To better capture semantic relationships, I transformed the operations into dense embeddings, via a language model fine-tuned for Ukrainian language. To generate dense representations of the operation texts for clustering, I experimented with two multilingual language models capable of producing embeddings: `'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'` and `'Alibaba-NLP/gte-multilingual-base'`. These models were chosen for their support of Ukrainian text and ability to capture semantic nuances in agricultural operation descriptions.
                
    I then applied `Principal Component Analysis` (PCA) to reduce dimensionality while preserving variance. Fitting `PCA` on the embeddings, I calculated the cumulative explained variance and identified the number of components needed for 95% variance retention. I visualized this with a plot showing the elbow point and a horizontal line at 0.95, then refit `PCA` with this optimal number, producing reduced DataFrame with columns named `'pca_1'`, `'pca_2'`, etc.
    
    Next, I experimented with clustering on these embeddings. For `KMeans`, I used the Elbow Method (plotting inertia) and Silhouette Score across a range of clusters (2 to 14) to find the optimal `k` (e.g., the maximum silhouette score). I selected and fitted KMeans, assigning cluster labels to training and validation sets as categorical features in DataFrame with a single column `'KMeans_operation_cluster'`.

    Similarly, I applied `DBSCAN` for density-based clustering, using parameters like `eps=0.5` and `min_samples=20`, to identify clusters and noise points without assuming spherical shapes. This yielded DataFrame with `'DBSCAN_operation_cluster'` column.

    `Result:` I incorporated these PCA components directly into the models and also used the cluster labels from KMeans on PCA-reduced data, as well as direct KMeans and DBSCAN on the embedding space. Despite these efforts, the results did not yield significant improvements in model performance, likely due to the sparse or noisy nature of the operation data. This led me to pivot toward even more complex methods.
    
    #### Final Approach: UMAP + HDBSCAN

    To address prior limitations, I optimized `Uniform Manifold Approximation and Projection` (UMAP) and `Hierarchical Density-Based Spatial Clustering of Applications with Noise` (HDBSCAN) on pre-computed operation embeddings using `Optuna`, maximizing the `Density-Based Clustering Validation` (DBCV) score for cluster quality.
    
    The Optuna objective embeddings_objective tuned `UMAP` parameters (`n_neighbors`: 10-30, `min_dist`: 0.0-0.25, `n_components`: 5-15, fixed `'cosine'` metric) and HDBSCAN (`min_cluster_size`: 10-50, `min_samples`: 5-15, `cluster_selection_method`: `'eom'` or `'leaf'`, Euclidean metric). `UMAP` reduced embeddings, then `HDBSCAN` clustered them; `DBCV` was computed, penalizing (<2 clusters) with -1. Ran `300 trials`. The best parameters with optimization results were saved in a json file.

    Fitted `UMAP` with optimal params to get reduced embeddings, then `HDBSCAN` for labels, noting clusters (excl. -1 noise) and noise count.

    Visualized with `2D UMAP` (n_components=2, metric='cosine', n_neighbors=15, min_dist=0.1): plotted labels, styling noise (smaller, gray, semi-transparent).
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image_paths['UMAP_HDBSCAN_2D'], caption=f"UMAP + HDBSCAN Results 2D Plot (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)", width='stretch')     
    with col2:
        st.image(image_paths['UMAP_HDBSCAN_2D_Alibaba'], caption=f"UMAP + HDBSCAN Results 2D Plot (Alibaba-NLP/gte-multilingual-base)", width='stretch')     
    
    st.markdown("""            
    Complementary `t-SNE` (n_components=2, perplexity=50, auto learning rate) plot, similar styling.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_paths['t-SNE'], caption=f"t-SNE Results 2D Plot (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)", width='stretch')     
    with col2:
        st.image(image_paths['t-SNE_Alibaba'], caption=f"t-SNE Results 2D Plot (Alibaba-NLP/gte-multilingual-base)", width='stretch')     
    
    st.markdown("""   
    `Result:` The `'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'` model produced more coherent and well-separated clusters in both UMAP and t-SNE plots, better aligning with expected patterns in the data. In contrast, `'Alibaba-NLP/gte-multilingual-base'` resulted in more diffuse or overlapping groups, which were less interpretable for this dataset. Consequently, I selected `'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'` for further processing and clustering. I saved the resulting feature in the categorical column `'HDBSCAN_operation_cluster'` and combined it with the main features.
      
    """)

    tab_idx += 1

with tabs[tab_idx]:

    st.markdown("""
    ### Pipeline
    To predict winter wheat yield, I implemented a pipeline using `LightGBM` for both point estimates (`mean`) and `quantile regression` (for 5% `lower` and 95% `upper` bounds, forming a 90% `confidence interval`). This approach allows for probabilistic forecasts, addressing the `bonus task`. Models were trained weekly from a starting week to the end week (simulating predictions 2 months before harvest onward), progressively excluding future features to evaluate accuracy over time. The pipeline incorporates `cross-validation` for robust metrics, holdout validation, visualizations, and `SHAP` explanations for feature importance.

    ### Hyperparameters
    I used the following parameters for all `LightGBM` models:
    - `n_estimators`: 1000
    - `learning_rate`: 0.01
    - `num_leaves`: 31
    - `max_depth`: 6
    - `min_child_samples`: 20
    - `min_child_weight`: 0.01
    - `subsample`: 0.7
    - `subsample_freq`: 1
    - `colsample_bytree`: 0.6
    - `reg_alpha`: 0.2
    - `reg_lambda`: 0.8

    For the `mean` model, the objective was `'regression'`. For quantiles, it was `'quantile'` with `alpha=0.05` for `lower` and `alpha=0.95` for `upper`.

    ### Training with Cross-Validation
    In the `train_lgbm_with_cv` function, I employed 5-fold Stratified K-Fold cross-validation, stratified by `'Region_cluster'` to maintain regional balance. For each fold:
    - Split data into train and validation sets.
    - Trained three models (`mean`, `lower`, `upper`) with early stopping (25 rounds) based on validation `MAPE`.
    - Computed fold metrics: `MAE`, `MAPE`, `R2` for mean; `Pinball loss` for quantiles.
    - Averaged metrics across folds for CV scores.

    After CV, I retrained three models for each week on the full training data for final use.

    ### Weekly Model Training
    To assess predictive accuracy over time, I iterated over weeks from `START_WEEK` (`20`) to `END_WEEK` (`28`):
    - For each week `w`, dropped features with `week_suffixes` > `w` (e.g., later NDVI or weather data).
    - Trained models using the above CV function on the training data.
    - Stored models and CV metrics in variable `results_weekly`.

    I visualized CV MAE and MAPE for the mean model across weeks using a dual-axis plot (saved as `'Model_Mean_accuracy_CV.png'`), showing how accuracy improves as more data becomes available closer to harvest.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image_paths['Model_Mean_accuracy_CV'], caption=f"Metrics for model `mean`", width='stretch')
    with col2:
        st.image(image_paths['Models_Lower_and_Upper_accuracy_CV'], caption=f"Metrics for models `lower` and `upper`", width='stretch')
    st.table(df_cv_metrics)
    
    st.markdown("""
    #### Mean Absolute Error (MAE)
    $$ \\text{MAE} = \\frac{1}{n} \\sum_{i=1}^{n} |y_i - \\hat{y}_i| $$

    MAE quantifies the average magnitude of errors between predicted $$ (\\hat{y}_i) $$ and actual $$ (y_i) $$ values, ignoring direction. It provides an intuitive measure of prediction accuracy in the target's units.

    #### Mean Absolute Percentage Error (MAPE)
    $$ \\text{MAPE} = \\frac{1}{n} \\sum_{i=1}^{n} \\left| \\frac{y_i - \\hat{y}_i}{y_i} \\right| \\times 100\\% $$

    MAPE expresses the average absolute error as a percentage of the actual values, making it useful for comparing relative errors across datasets or models, though sensitive to near-zero actuals.

    #### Coefficient of Determination (R²)
    $$ R^2 = 1 - \\frac{\\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2}{\\sum_{i=1}^{n} (y_i - \\bar{y})^2} $$

    R² indicates how well the model explains the variance in the data, with values closer to 1 signifying better fit (where $$ \\bar{y} $$ is the mean of actuals). It can be negative if worse than a mean baseline.

    #### Pinball Loss (Quantile Loss)
    For a quantile $$ (\\alpha) $$ :
    $$ \\text{Pinball Loss} = \\frac{1}{n} \\sum_{i=1}^{n} \\max(\\alpha (y_i - \\hat{y}_i), (\\alpha - 1)(y_i - \\hat{y}_i)) $$

    Pinball loss is used in quantile regression to evaluate predictions at specific quantiles.
    
    Values of `α < 0.5` make overprediction much more expensive. As such, the model is incentivized to underpredict the target.
    
    Values of `α > 0.5` make underprediction much more expensive. As such, the model is incentivized to overpredict the target.
    
    ### Configuration Saving
    Finally, I saved a training configuration JSON with the random seed, embedding model name, feature list, and prediction weeks range for reproducibility.
    """)

    tab_idx += 1

with tabs[tab_idx]:
    # st.header("Eval")

    st.markdown("""   
    ### Validation and Predictions
    On the holdout validation set:
    - For each week, filtered features similarly.
    - Predicted with the final models.
    - Computed validation metrics (`MAE`, `MAPE`, `R2` for mean; `Pinball` for quantiles).
    """)
    
    st.table(df_val_metrics)   
    
    st.markdown("""   
    ### Model Interpretability with SHAP
    To identify influential features, I computed `SHAP` values for each model type (`mean`, `lower`, `upper`) per week using `TreeExplainer` on the validation set.
    """)

    st.subheader('Select week:')
    week_options = train_coinfig['pred_weeks_range']
    selected_week = st.pills('Select week:', week_options, selection_mode='single', label_visibility='hidden', default=week_options[0])
    
    st.subheader('SHAP values for `Lower`, `Mean` and `Upper` models:')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(shap_path / f'lower_week_{selected_week}.png')
    with col2:
        st.image(shap_path / f'mean_week_{selected_week}.png')
    with col3:
        st.image(shap_path / f'upper_week_{selected_week}.png')

    st.subheader('Forecasts with 5% and 95% confidence intervals:')
    st.image(confidence_intervals_path / f'week_{selected_week}.png')

    tab_idx += 1

with tabs[tab_idx]:

    import pandas as pd
    from lightgbm import LGBMRegressor
    import shap 

    st.markdown("""
    Upload new data (similar to the training data format, e.g., parquet with features like weather, NDVI, operations) for yield prediction.
    The files will be preprocessed through the pipeline (cleaning, embeddings, clustering), and predictions will be made using the trained models for a selected week.
    """)

    uploaded_main_file = st.file_uploader("Upload new main file", type=['parquet'])
    uploaded_weather_file = st.file_uploader("Upload new operations file", type=['parquet'])

    if (uploaded_main_file is not None) and (uploaded_weather_file is not None):
        df_main = pd.read_parquet(uploaded_main_file)
        df_operations = pd.read_parquet(uploaded_weather_file)
        st.success("Files uploaded successfully!")
    else:
        st.info("Please upload files to make predictions.")

    tab_idx += 1


