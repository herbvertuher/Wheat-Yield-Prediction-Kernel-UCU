# %%
# Load libs and datasets
from pathlib import Path
import re
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import shap
import optuna
from scipy.cluster.hierarchy import dendrogram, linkage
from pyproj import Transformer

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_pinball_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE

import umap
import hdbscan
from hdbscan.validity import validity_index
from sentence_transformers import SentenceTransformer

from lightgbm import LGBMRegressor
from lightgbm.callback import early_stopping

GLOBAL_RANDOM_SEED = 42
# GLOBAL_RANDOM_SEED = np.random.randint(10000)
np.random.seed(GLOBAL_RANDOM_SEED)
print('GLOBAL_RANDOM_SEED =', GLOBAL_RANDOM_SEED)

LOWER_QUANTILE = 0.05
UPPER_QUANTILE = 0.95

START_WEEK = 20
END_WEEK = 28

root_path = Path(__file__).resolve().parents[0]

data_path = root_path / 'data'
artifacts_path = root_path / 'artifacts'
plots_path = root_path / 'plots'
metrics_path = root_path / 'metrics'
confidence_intervals_path = plots_path / 'confidence_intervals'
shap_path = plots_path / 'shap'

for directory in [artifacts_path, plots_path, metrics_path, confidence_intervals_path, shap_path]:
    directory.mkdir(parents=True, exist_ok=True)

df_main = pd.read_parquet(data_path / 'df_2025_v2_extended_weather.parquet')
df_operations = pd.read_parquet(data_path / 'operations_2025.parquet')

# initial preparation `main` dataframe
df_main = df_main[df_main['Culture'].notna()]
df_main = df_main.loc[:, ~df_main.columns.str.contains('_prev_year')]
suffix = df_main.columns.str.extract(r'(\d+)$')[0].astype(float)
selected_cols = df_main.columns[suffix >= 34]
df_main = df_main.drop(columns=selected_cols)
df_main = df_main[df_main['Area'].notna()]
df_main = df_main.drop(columns=['Year', 'Culture', 'Moisture', 
                                'WeatherGridId', 
                                'crop', 
                                'County', 
                                'Mex', 
                                'Cluster'
                                ])

col_patterns_to_drop = [
                        'CumSum_Precipitation',
                        'CumSum_TempEffective',
                        'WindSpeed',
                        'Sunshine_duration_avg',
                        'TempStandardMin',
                        'TempStandardMax',
                        ]
cols_to_drop = [column for column in df_main.columns
                  if any(substring in column for substring in col_patterns_to_drop)]
df_main = df_main.drop(columns=cols_to_drop)

# initial preparation `operations` dataframe
# Перейменування колонок
df_operations.columns = ['operation', 'detailed_operation', 'year_of_start', 'date_of_start', 'total', 'field_id']
df_operations['date_of_start'] = pd.to_datetime(df_operations['date_of_start'])

# operation_column = 'operation' 
# ===== \\\ OR //// =====
operation_column = 'detailed_operation'

# %%

# Merge duplicated rows if delta in operations time is lower that `gap_in_days`
gap_in_days = 30
# Список текстових колонок для групування
group_cols = ['field_id', 'operation', 'detailed_operation']

def merge_close_dates(group: pd.DataFrame) -> pd.DataFrame:
    # Сортуємо всередині групи
    group = group.sort_values('date_of_start').reset_index(drop=True)

    merged_rows = []
    current_start = group.loc[0, 'date_of_start']
    current_total = group.loc[0, 'total']

    for i in range(1, len(group)):
        delta = (group.loc[i, 'date_of_start'] - group.loc[i-1, 'date_of_start']).days
        if delta <= gap_in_days:
            # Якщо дата близька – додаємо total
            current_total += group.loc[i, 'total']
        else:
            # Закриваємо попередній блок
            merged_rows.append({
                'date_of_start': current_start,
                'total': current_total
            })
            # Починаємо новий блок
            current_start = group.loc[i, 'date_of_start']
            current_total = group.loc[i, 'total']

    # Додаємо останній блок
    merged_rows.append({
        'date_of_start': current_start,
        'total': current_total
    })

    return pd.DataFrame(merged_rows)

# Застосовуємо до всіх груп
df_operations_merged = (
    df_operations
    .groupby(group_cols, group_keys=True)
    .apply(merge_close_dates)
    .reset_index().drop(columns=['level_3'])
)
df_operations_merged[operation_column] = df_operations_merged[operation_column].str.replace('.', '')

df_operations_merged_rows = df_operations_merged.groupby('field_id', as_index=False)[operation_column].apply(lambda x: '. '.join(x))

# %%

# Обчислюємо кількість операцій у кожному рядку
operation_counts = df_operations_merged_rows['detailed_operation'].apply(lambda row: len(row.split('.')))
# Підраховуємо частоти
count_series = operation_counts.value_counts().sort_index()
# Перетворюємо у відсотки
percentage = (count_series / count_series.sum()) * 100

plt.figure(figsize=(10, 6), dpi=300)
plt.bar(count_series.index, percentage, color='steelblue', edgecolor='black', alpha=0.85)
for i, (x, y) in enumerate(zip(count_series.index, percentage)):
    plt.text(x, y + 0.5, f'{y:.1f}%', ha='center', va='bottom', fontsize=9)
plt.xlabel('Number of operations', fontsize=12)
plt.ylabel('Percentage', fontsize=12)
plt.title('Distribution of the number of operations per field')
plt.xticks(percentage.index)
plt.tight_layout()
plt.savefig(plots_path / 'Distribution_number_of_operations.png', dpi=300)
plt.show()

# %%

# Перетворення тексту в ембедінги
embedding_models = [
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    # 'Alibaba-NLP/gte-multilingual-base',
]
selected_embedding_model_name = embedding_models[0]

sentences_to_embeddings = df_operations_merged_rows[operation_column].to_list()

model = SentenceTransformer(selected_embedding_model_name, trust_remote_code=True)
embeddings = model.encode(sentences_to_embeddings, normalize_embeddings=True)

df_embeddings = pd.DataFrame(data=embeddings,
                             index=df_operations_merged_rows['field_id'],
                             columns=[f'embedding_{i+1}' for i in range(embeddings.shape[1])])

print('Embeddings shape:', embeddings.shape)

# %%

def embeddings_objective(trial):
    # Гіперпараметри UMAP
    n_neighbors = trial.suggest_int('umap_n_neighbors', 10, 30)
    min_dist = trial.suggest_float('umap_min_dist', 0.0, 0.25)
    n_components = trial.suggest_int('umap_n_components', 5, 15)
    
    # Гіперпараметри HDBSCAN
    min_cluster_size = trial.suggest_int('hdbscan_min_cluster_size', 10, 50) 
    min_samples = trial.suggest_int('hdbscan_min_samples', 5, 15)
    cluster_selection_method = trial.suggest_categorical('hdbscan_cluster_selection_method', ['eom', 'leaf'])
    
    # Застосування UMAP
    umap_reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric='cosine',  # Фіксований для семантичних ембедінгів
        random_state=GLOBAL_RANDOM_SEED
    )
    embeddings_reduced = umap_reducer.fit_transform(embeddings)
    embeddings_reduced = embeddings_reduced.astype(np.float64)  # Щоб уникнути помилки в validity_index
    
    # Застосування HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        metric='euclidean',  # Для зменшених даних
        gen_min_span_tree=True
    )
    labels = clusterer.fit_predict(embeddings_reduced)
   
    # Обчислення DBCV (validity_index)
    if len(np.unique(labels)) > 1:  # Потрібно щонайменше 2 кластери
        score = validity_index(embeddings_reduced, labels, metric='euclidean')
    else:
        score = -1  # Погана оцінка, якщо немає кластерів
    
    return score  # Optuna максимізує цю функцію (DBCV -> max=1)

best_params_filename = 'optuna_umap_hdbscan_best_params.json'
if (artifacts_path / best_params_filename).exists():
    with open(artifacts_path / best_params_filename, 'r', encoding='utf-8') as f:
        embedding_best_params = json.load(f)
    print(f"Параметри завантажено з {best_params_filename}")
else:
    # Запускаємо оптимізацію з Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(embeddings_objective, n_trials=300)  # Максимізуємо DBCV
    embedding_best_params = study.best_params
    print("Найкращі параметри:", embedding_best_params)

    # Збереження в JSON
    with open(artifacts_path / best_params_filename, 'w') as f:
        json.dump(embedding_best_params, f, indent=4)
    print(f"Параметри збережено в {best_params_filename}")

# %%

# Зчитування параметрів для UMAP
umap_params = {
    'n_neighbors': embedding_best_params['umap_n_neighbors'],
    'min_dist': embedding_best_params['umap_min_dist'],
    'n_components': embedding_best_params['umap_n_components'],
    'metric': 'cosine',  # Фіксований
    'random_state': GLOBAL_RANDOM_SEED,   # Для відтворюваності
}

# Зчитування параметрів для HDBSCAN
hdbscan_params = {
    'min_cluster_size': embedding_best_params['hdbscan_min_cluster_size'],
    'min_samples': embedding_best_params['hdbscan_min_samples'],
    'cluster_selection_method': embedding_best_params['hdbscan_cluster_selection_method'],
    'metric': 'euclidean',  # Фіксований
    'prediction_data': True,
}

# Зменшення розмірності з UMAP (для кластеризації)
umap_reducer = umap.UMAP(**umap_params)
embeddings_reduced = umap_reducer.fit_transform(embeddings)

# Кластеризація з HDBSCAN
clusterer = hdbscan.HDBSCAN(**hdbscan_params)
labels = clusterer.fit_predict(embeddings_reduced)

# Кількість кластерів (ігноруючи шум -1)
n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
print(f"Found {n_clusters} clusters. Noise: {np.sum(labels == -1)} points.")

# Крок 3: Візуалізація (2D UMAP для перегляду)
umap_2d = umap.UMAP(n_components=2, metric='cosine', n_neighbors=15, min_dist=0.1, random_state=GLOBAL_RANDOM_SEED)
embeddings_2d = umap_2d.fit_transform(embeddings)

plt.figure(figsize=(12, 8))
unique_labels = np.unique(labels)
n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Кількість кластерів без шуму
colors = sns.color_palette('husl', n_colors=len(unique_labels))

for i, lab in enumerate(unique_labels):
    mask = labels == lab
    if not np.any(mask):  # Перевірка на порожній кластер
        continue
    label_name = 'Noise' if lab == -1 else f'Cluster {lab}'
    # Виділяємо шум: менший розмір, сірий колір, менша прозорість
    size = 20 if lab == -1 else 40
    color = 'gray' if lab == -1 else colors[i]
    alpha = 0.5 if lab == -1 else 1.0
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], s=size, alpha=alpha, 
                color=color, label=label_name, edgecolor='k', linewidth=0.5)  # Обводка для видимості

# Легенда замість colorbar: краща для дискретних міток
plt.legend(title='Cluster label:', bbox_to_anchor=(1.01, 1), loc='upper left', 
           borderaxespad=0, shadow=True, fontsize='medium')

plt.title('Clustering embeddings (UMAP + HDBSCAN)', fontsize=16)
plt.xlabel('Component 1', fontsize=12)
plt.ylabel('Component 2', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(plots_path / 'umap_hdbscan_2D_visualization.png', dpi=300)
plt.show()

# # Опціонально: Аналіз кластерів (вивести приклади текстів)
# unique_labels = np.unique(labels)
# for label in unique_labels:
#     if label != -1:  # Ігнорувати шум
#         cluster_texts = [sentences_to_embeddings[i] for i in range(len(labels)) if labels[i] == label]
#         print(f"Кластер {label}: {len(cluster_texts)} текстів. Приклади: {cluster_texts[:3]}")

# %%

# Використання t-SNE
tsne = TSNE(n_components=2, perplexity=50, learning_rate='auto', random_state=GLOBAL_RANDOM_SEED)
embeddings_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(12, 8))
unique_labels = np.unique(labels)
n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Кількість кластерів без шуму
colors = sns.color_palette('husl', n_colors=len(unique_labels))

cmap = ListedColormap(colors)

for i, lab in enumerate(unique_labels):
    mask = (labels == lab)
    if not np.any(mask):  # Перевірка на порожній кластер (краща практика)
        continue
    label_name = 'Noise' if lab == -1 else f'Cluster {lab}'
    # Для шуму використовуємо менший розмір і сірий колір (якщо lab == -1)
    size = 20 if lab == -1 else 40
    color = 'gray' if lab == -1 else cmap(i)
    alpha = 0.5 if lab == -1 else 1.0  # Менша прозорість для шуму
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], s=size, alpha=alpha, 
                color=color, label=label_name, edgecolor='k', linewidth=0.5)  # Додаємо обводку для кращої видимості

plt.legend(title='Cluster label:', bbox_to_anchor=(1.01, 1), loc='upper left', 
           borderaxespad=0, shadow=True, fontsize='medium')

plt.title('Visualization of embeddings using t-SNE', fontsize=16)
plt.xlabel('Component 1', fontsize=12)
plt.ylabel('Component 2', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(plots_path / 't-SNE_visualization.png', dpi=300)
plt.show()

df_embeddings_labels = pd.DataFrame(data=labels,
                                    index=df_operations_merged_rows['field_id'],
                                    columns=['HDBSCAN_operation_cluster']
                                    ).reset_index(drop=False)
df_embeddings_labels['HDBSCAN_operation_cluster'] = \
        df_embeddings_labels['HDBSCAN_operation_cluster'].astype('category')

# %%

df_main_prepared = (df_main
                    .merge(df_embeddings_labels, left_on='Field_id', right_on='field_id')
                    .drop(columns=['Field_id'])
                    .set_index('field_id')
                   )

# %%

linkage_method = 'ward'
selected_n_clusters = 4

# трансофрмація Latitude і Longitude в локальні координати для кращої візуалізації
transformer = Transformer.from_crs("epsg:4326", "epsg:32636", always_xy=True)
df_main_prepared['coordinate_x'], df_main_prepared['coordinate_y'] = transformer.transform(df_main_prepared['Longitude'].to_numpy(), df_main_prepared['Latitude'].to_numpy())

df_main_prepared['coordinate_x'] = df_main_prepared['coordinate_x'] / 1000
df_main_prepared['coordinate_y'] = df_main_prepared['coordinate_y'] / 1000

coord_scaler = StandardScaler()
coordinates_scaled = coord_scaler.fit_transform(df_main_prepared[['coordinate_x', 'coordinate_y']])

# Метод 'ward' мінімізує дисперсію всередині кластерів при їх об'єднанні.
# Це найпопулярніший та найчастіше рекомендований метод.
linked = linkage(coordinates_scaled, method=linkage_method)

plt.figure(figsize=(14, 7))
dendrogram(linked,
            orientation='top',
            labels=None, # можна передати мітки для кожної точки
            distance_sort='descending',
            show_leaf_counts=True)
plt.title('Hierarchical Clustering (Dendrogram)')
plt.xticks([])
plt.ylabel('Euclidean Distance')
plt.savefig(plots_path / 'Dendrogram_visualization.png', dpi=300)
plt.show()

agg_cluster = AgglomerativeClustering(n_clusters=selected_n_clusters, linkage=linkage_method, metric='euclidean')  # ward
region_labels = agg_cluster.fit_predict(coordinates_scaled)

plt.figure(figsize=(10, 7))
plt.scatter(coordinates_scaled[:, 0], coordinates_scaled[:, 1], c=region_labels, cmap='viridis', s=50)
plt.title('Agglomerative Clustering Results (n_clusters=4)')
plt.xlabel('Scaled coordinate_x')
plt.ylabel('Scaled coordinate_y')
plt.grid(True)
plt.savefig(plots_path / 'AgglomerativeClustering_results_visualization.png', dpi=300)
plt.show()

# Альтернативне використання KMeans для кластеризації
# kmeans_region = KMeans(n_clusters=4, random_state=GLOBAL_RANDOM_SEED)
# region_labels = kmeans_region.fit_predict(coordinates_scaled)

df_main_prepared['Region_cluster'] = region_labels
df_main_prepared['Region_cluster'] = df_main_prepared['Region_cluster'].astype('category')

# sns.scatterplot(data=df_main_prepared, x='coordinate_x', y='coordinate_y', hue='Region_cluster', palette='viridis', s=50)
# sns.scatterplot(data=df_main_prepared, x='coordinate_x', y='coordinate_y', hue='WeatherGridId', palette='viridis')

df_main_prepared = df_main_prepared.drop(columns=[
                                                  'Longitude', 'Latitude', 
                                                  'coordinate_x', 'coordinate_y',
                                                  ])

# %%

X = df_main_prepared.drop(columns=['Yield'])
y = df_main_prepared['Yield']

# %%

X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=0.15, stratify=X['Region_cluster'], random_state=GLOBAL_RANDOM_SEED)

# %%

X_trn_prep = X_trn.copy()
X_val_prep = X_val.copy()

print("Final shape:", X_trn_prep.shape)

# %%

# Model cycle
lgbm_params = {
    "n_estimators": 1000,
    "learning_rate": 0.01,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "min_child_weight": 1e-2,
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.2,
    "reg_lambda": 0.8,
    "random_state": GLOBAL_RANDOM_SEED,
    "n_jobs": -1
}

def train_lgbm_with_cv(X, y, stratify_col_name, params, n_splits=5):
    stratify_col_data = X[stratify_col_name]
    X = X.drop(columns=[stratify_col_name])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=GLOBAL_RANDOM_SEED)

    mae_mean = []
    mape_mean = []
    r2_mean = []

    pinball_lower = []
    pinball_upper = []

    def init_and_fit_models(X_train, y_train, cv_mode: bool, X_eval=None, y_eval=None):
        model_mean = LGBMRegressor(objective='regression', **params)
        model_lower = LGBMRegressor(objective='quantile', alpha=LOWER_QUANTILE, **params)
        model_upper = LGBMRegressor(objective='quantile', alpha=UPPER_QUANTILE, **params)

        for model in [model_mean, model_lower, model_upper]:
            if cv_mode:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_eval, y_eval)],
                    eval_metric="mape",
                    callbacks=[early_stopping(stopping_rounds=25, verbose=True)],
                )
            else:
                model.fit(
                    X_train, y_train,
                    eval_metric=mean_absolute_error,
                )
        return model_mean, model_lower, model_upper

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, stratify_col_data), 1):

        # Split data with CV
        X_train, X_eval = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_eval = y.iloc[train_idx], y.iloc[val_idx]

        # Initialize models for fold
        model_mean_cv, model_lower_cv, model_upper_cv = init_and_fit_models(X_train, y_train, cv_mode=True, X_eval=X_eval, y_eval=y_eval)

        # Generate fold predictions
        y_pred_mean = model_mean_cv.predict(X_eval)
        y_pred_lower = model_lower_cv.predict(X_eval)
        y_pred_upper = model_upper_cv.predict(X_eval)

        # CV metrics for `mean` model
        mae_mean.append(mean_absolute_error(y_eval, y_pred_mean))
        mape_mean.append(mean_absolute_percentage_error(y_eval, y_pred_mean))
        r2_mean.append(r2_score(y_eval, y_pred_mean))

        # CV metrics for `lower` model
        pinball_lower.append(mean_pinball_loss(y_eval, y_pred_lower, alpha=LOWER_QUANTILE))

        # CV metrics for `upper` model
        pinball_upper.append(mean_pinball_loss(y_eval, y_pred_upper, alpha=UPPER_QUANTILE))

    # Train models for all data without CV
    model_mean, model_lower, model_upper = init_and_fit_models(X, y, cv_mode=False)

    # Summarize metrics for `mean` model
    mean_mae_cv = np.round(np.mean(mae_mean), 4)
    mean_mape_cv = np.round(np.mean(mape_mean), 4)
    mean_r2_cv = np.round(np.mean(r2_mean), 4)

    # Summarize metrics for `lower` model
    lower_pinball_cv = np.round(np.mean(pinball_lower), 4)

    # Summarize metrics for `upper` model
    upper_pinball_cv = np.round(np.mean(pinball_upper), 4)

    print(f"\nCV MAE on Train: {mean_mae_cv:.4f} ± {np.std(mae_mean):.4f}")
    print(f"CV MAPE on Train: {mean_mape_cv:.4f} ± {np.std(mape_mean):.4f}")

    return {
        'models': {'mean': model_mean, 'lower': model_lower, 'upper': model_upper},
        'mean_model_metrics': {'cv_mae': mean_mae_cv, 'cv_mape': mean_mape_cv, 'cv_r2': mean_r2_cv},
        'lower_model_metrics': {'cv_pinball': lower_pinball_cv},
        'upper_model_metrics': {'cv_pinball': upper_pinball_cv},
    }

# %%

# Основний цикл з навчанням моделей за збереженням CV метрик
results_weekly = {}
for w in range(START_WEEK, END_WEEK+1):
    print(w)
    suffix = X_trn_prep.columns.str.extract(r'(\d+)$')[0].astype(float)
    week_cols_to_drop = X_trn_prep.columns[suffix > w]
    X_trn_week = X_trn_prep.drop(columns=week_cols_to_drop)
    results_weekly[w] = train_lgbm_with_cv(X_trn_week, y_trn, 'Region_cluster', lgbm_params, n_splits=5)

# %%

# Створення таблиці з CV метриками
data = []
for w in results_weekly.keys():
    row = {
        'cv_mae_mean': results_weekly[w]['mean_model_metrics']['cv_mae'],
        'cv_mape_mean': results_weekly[w]['mean_model_metrics']['cv_mape'],
        'cv_r2_mean': results_weekly[w]['mean_model_metrics']['cv_r2'],
        'cv_pinball_lower': results_weekly[w]['lower_model_metrics']['cv_pinball'],
        'cv_pinball_upper': results_weekly[w]['upper_model_metrics']['cv_pinball'],
    }
    data.append(row)

df_cv_metrics = pd.DataFrame(data, index=results_weekly.keys())
df_cv_metrics.to_pickle(metrics_path / 'df_cv_metrics.pkl')
df_cv_metrics

# %%

weeks = results_weekly.keys()


# Візуалізація CV метрик моделі `mean` по тижнях
fig, ax1 = plt.subplots(figsize=(10, 6))

# Перша вісь (MAE)
ax1.plot(weeks, df_cv_metrics['cv_mae_mean'], marker='o', color='blue', label='MAE `mean`')
ax1.set_xlabel('Week')
ax1.set_ylabel('CV MAE', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_yticks(np.arange(0.5, 0.8, 0.05))
ax1.grid(True, which='both', linestyle='--', alpha=0.5)

# Друга вісь (MAPE)
ax2 = ax1.twinx()
ax2.plot(weeks, df_cv_metrics['cv_mape_mean'], marker='s', color='purple', label='MAPE `mean`')
ax2.set_ylabel('CV MAPE', color='purple')
ax2.set_yticks(np.arange(0.08, 0.14, 0.01))
ax2.tick_params(axis='y', labelcolor='purple')

plt.title('Accuracy of the `mean` model by week')
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

plt.tight_layout()
plt.savefig(plots_path / f'Model_Mean_accuracy_CV.png', dpi=300)
plt.show()

# %%

# Візуалізація CV метрик моделей `lower` та `upper` по тижнях
fig, ax1 = plt.subplots(figsize=(10, 6))

# Перша вісь (Pinball Lower)
ax1.plot(weeks, df_cv_metrics['cv_pinball_lower'], marker='o', color='blue', label='Pinball `lower`')
ax1.set_xlabel('Week')
ax1.set_ylabel('CV Pinball Loss', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_yticks(np.arange(0.08, 0.14, 0.01))
ax1.grid(True, which='both', linestyle='--', alpha=0.5)

# Друга вісь (Pinball Upper)
ax2 = ax1.twinx()
ax2.plot(weeks, df_cv_metrics['cv_pinball_upper'], marker='s', color='purple', label='Pinball `upper`')
ax2.set_ylabel('CV Pinball Loss', color='purple')
ax2.set_yticks(np.arange(0.08, 0.14, 0.01))
ax2.tick_params(axis='y', labelcolor='purple')

# Заголовок і легенда
plt.title('Accuracy of the `lower` and `upper` models by week')
# Для обох осей одночасно
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

# Збереження та показ
plt.tight_layout()
plt.savefig(plots_path / f'Models_Lower_and_Upper_accuracy_CV.png', dpi=300)
plt.show()


# %%

# Створення таблиці з метриками на валідаційній вибірці та збереження прогнозів
val_metrics = []
y_val_preds_weekly = {k: {} for k in results_weekly.keys()}
for w in results_weekly.keys():
    week_cols_to_drop = X_val_prep.columns[X_val_prep.columns.str.extract(r'(\d+)$')[0].astype(float) > w]
    X_val_week = X_val_prep.drop(columns=week_cols_to_drop)
    X_val_week = X_val_week.drop(columns=['Region_cluster'])
    row = {}
    for model_type in ['mean', 'lower', 'upper']:
        model = results_weekly[w]['models'][model_type]
        y_val_pred = model.predict(X_val_week)
        y_val_preds_weekly[w][model_type] = y_val_pred  # Збереження прогнозів для візуалізації
        if model_type == 'mean':
            row[f'val_mae_{model_type}'] = np.round(mean_absolute_error(y_val, y_val_pred), 4)
            row[f'val_mape_{model_type}'] = np.round(mean_absolute_percentage_error(y_val, y_val_pred), 4)
            row[f'val_r2_{model_type}'] = np.round(r2_score(y_val, y_val_pred), 4)
        else:
            current_quantile = LOWER_QUANTILE if model_type == 'lower' else UPPER_QUANTILE
            row[f'val_pinball_{model_type}'] = np.round(mean_pinball_loss(y_val, y_val_pred, alpha=current_quantile), 4)
        
    val_metrics.append(row)

df_val_metrics = pd.DataFrame(val_metrics, index=results_weekly.keys())
df_val_metrics.to_pickle(metrics_path / 'df_val_metrics.pkl')
df_val_metrics

# %% 

for w in y_val_preds_weekly.keys():
    df_to_plot = pd.concat([y_val.reset_index(drop=False), pd.DataFrame(y_val_preds_weekly[w])], axis=1).set_index(['field_id'])
    df_to_plot = df_to_plot.sort_values(['Yield'])
    df = df_to_plot.copy()

    x = df.index.astype(str)
    plt.figure(figsize=(20, 6), dpi=300)
    plt.plot(x, df['Yield'], color='green', marker='*', linestyle='', alpha=0.5, label='Actual value')
    plt.plot(x, df['mean'],  color='blue', marker='*',  linestyle='', alpha=0.5, label='Predicted value')
    plt.plot(x, df['lower'], color='purple', marker='', linestyle='--', label='Lower bound')
    plt.plot(x, df['upper'], color='brown', marker='', linestyle='--', label='Upper bound')
    plt.xlabel('Field ID')
    plt.ylabel('Yield (t/ha)')
    plt.xticks(rotation=90)
    plt.title(f'Visualization of forecasts for {w} weeks with confidence intervals of 5% and 95%')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(confidence_intervals_path / f'week_{w}.png', dpi=300)
    plt.show()

# %%

# Розрахунок SHAP values для моделей кожного тижня
model_types = [
    'mean', 
    'lower', 
    'upper'
]
model_to_show = 'mean'

# Побудова графіків SHAP важливості ознак для кожного тижня та моделі
for w in results_weekly.keys():
    week_cols_to_drop = X_val_prep.columns[X_val_prep.columns.str.extract(r'(\d+)$')[0].astype(float) > w]
    X_val_week = X_val_prep.drop(columns=week_cols_to_drop)
    X_val_week = X_val_week.drop(columns=['Region_cluster'])
    for model_type in model_types:
        model = results_weekly[w]['models'][model_type]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val_week)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_val_week, show=False)
        plt.title(f'SHAP  |  Week: `{w}`  |  Model: `{model_type.capitalize()}`')
        plt.tight_layout()
        plt.savefig(shap_path / f'{model_type}_week_{w}.png', dpi=300)
        if model_type == model_to_show:
            plt.show()
        else:
            plt.close()

# %%

# Save train_coinfig.json
train_coinfig = {
    'random_seed': GLOBAL_RANDOM_SEED,
    'embedding_model_name': selected_embedding_model_name,
    'features': X.columns.to_list(),
    'pred_weeks_range': list(range(START_WEEK, END_WEEK+1))
}
with open(artifacts_path / 'train_config.json', 'w') as f:
    json.dump(train_coinfig, f, indent=4)

# %%