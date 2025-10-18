# %%
# Load libs and datasets
import os
os.chdir('D:/Projects/Kernel_Yield_Prediction_UCU/')

import warnings

import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import optuna
from scipy.sparse import hstack, csr_matrix
from scipy.cluster.hierarchy import dendrogram, linkage
from pyproj import Transformer

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import KNNImputer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE

import umap
import hdbscan
# from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from lightgbm import LGBMRegressor
from lightgbm.callback import early_stopping
# from xgboost import XGBRegressor

# GLOBAL_RANDOM_SEED = 8
GLOBAL_RANDOM_SEED = np.random.randint(10000)
np.random.seed(GLOBAL_RANDOM_SEED)
print('GLOBAL_RANDOM_SEED =', GLOBAL_RANDOM_SEED)

use_bow = False
use_tfidf = False
use_pca = False
use_kmeans = False
use_dbscan = False

# Remove Warnings
# warnings.filterwarnings('ignore')
# Disable LightGBM warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", message="[LightGBM] [Warning] No further splits with positive gain, best gain: -inf")

# df_main_old = pd.read_parquet('data/df_2025.parquet')
df_main = pd.read_parquet('data/df_2025_v2_extended_weather.parquet')
df_operations = pd.read_parquet('data/operations_2025.parquet')

# initial preparation `main` dataframe
df_main = df_main[df_main['Culture'].notna()]
df_main = df_main.loc[:, ~df_main.columns.str.contains('_prev_year')]
suffix = df_main.columns.str.extract(r'(\d+)$')[0].astype(float)
selected_cols = df_main.columns[suffix >= 34]
df_main = df_main.drop(columns=selected_cols)
df_main = df_main[df_main['Area'].notna()]
df_main = df_main.drop(columns=['Year', 'Culture', 'Moisture', 
                                'WeatherGridId', 
                                # 'K',
                                # 'Area',
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
                        # 'EvapoTranspiration',
                        # 'PrecipitationCumulative_sum',
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

umap_params = {
    "n_neighbors": 30,
    "n_components": 10,
    "min_dist": 0.1,
    "metric": "cosine",
    "random_state": GLOBAL_RANDOM_SEED
}

hdbscan_params = {
    "min_cluster_size": 30,
    "min_samples": 5,
    "metric": "euclidean",
    "cluster_selection_method": "eom"
}

# Зменшення розмірності з UMAP (для кластеризації)
umap_reducer = umap.UMAP(**umap_params)
embeddings_reduced = umap_reducer.fit_transform(embeddings)

# Кластеризація з HDBSCAN
clusterer = hdbscan.HDBSCAN(**hdbscan_params)
labels = clusterer.fit_predict(embeddings_reduced)

# Кількість кластерів (ігноруючи шум -1)
n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
print(f"Знайдено {n_clusters} кластерів. Шум: {np.sum(labels == -1)} точок.")

# Крок 3: Візуалізація (2D UMAP для перегляду)
umap_2d = umap.UMAP(n_components=2, metric='cosine', n_neighbors=15, min_dist=0.1, random_state=GLOBAL_RANDOM_SEED)
embeddings_2d = umap_2d.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='Spectral', alpha=0.7)
plt.colorbar()
plt.title('Кластеризація ембедінгів (UMAP + HDBSCAN)')
plt.xlabel('Компонента 1')
plt.ylabel('Компонента 2')
plt.grid(True)
plt.show()

# # Опціонально: Аналіз кластерів (вивести приклади текстів)
# unique_labels = np.unique(labels)
# for label in unique_labels:
#     if label != -1:  # Ігнорувати шум
#         cluster_texts = [sentences_to_embeddings[i] for i in range(len(labels)) if labels[i] == label]
#         print(f"Кластер {label}: {len(cluster_texts)} текстів. Приклади: {cluster_texts[:3]}")

# %%

# Використання t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Візуалізація (додайте labels, якщо є)
plt.figure(figsize=(10, 8))
unique_labels = np.unique(labels)
for lab in unique_labels:
    mask = labels == lab
    label_name = 'noise' if lab == -1 else f'cluster {lab}'
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], s=40, alpha=0.7, label=label_name)
plt.legend(title='Labels', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Візуалізація ембедінгів за допомогою t-SNE')
plt.xlabel('Компонента 1')
plt.ylabel('Компонента 2')
plt.grid(True)
plt.show()


df_embeddings_labels = pd.DataFrame(data=labels,
                                    index=df_operations_merged_rows['field_id'],
                                    columns=['HDBSCAN_cluster_label']
                                    ).reset_index(drop=False)
df_embeddings_labels['HDBSCAN_cluster_label'] = \
        df_embeddings_labels['HDBSCAN_cluster_label'].astype('category')

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

plt.figure(figsize=(12, 7))
dendrogram(linked,
            orientation='top',
            labels=None, # можна передати мітки для кожної точки
            distance_sort='descending',
            show_leaf_counts=True)
plt.title('Ієрархічна кластеризація (Дендрограма)')
plt.xlabel('Індекс точки даних')
plt.ylabel('Відстань (Евклідова)')
plt.show()

agg_cluster = AgglomerativeClustering(n_clusters=selected_n_clusters, linkage=linkage_method, metric='euclidean')  # ward
region_labels = agg_cluster.fit_predict(coordinates_scaled)

plt.figure(figsize=(10, 7))
plt.scatter(coordinates_scaled[:, 0], coordinates_scaled[:, 1], c=region_labels, cmap='viridis', s=50)
plt.title('Результати агломеративної кластеризації (n_clusters=4)')
plt.xlabel('coordinate_x (масштабована)')
plt.ylabel('coordinate_y (масштабована)')
plt.grid(True)
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

# заповнюємо пропуски, окільки є декілька пропусків в `P`
# /// Вже не актуально. Заповнення не потрібне 
# imputer = KNNImputer(n_neighbors=5).set_output(transform='pandas')
# X = imputer.fit_transform(X)

# %%

X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=0.15, stratify=X['Region_cluster'], random_state=GLOBAL_RANDOM_SEED)

# Для роботи старого коду з PCA та KMeans
trn_embeddings = df_embeddings.loc[X_trn.index, :]
val_embeddings = df_embeddings.loc[X_val.index, :]

# %%

if use_bow:
    # ============== Bag of Words ==============
    def custom_preprocessor(text):
        # Прибираємо дужки та крапки
        text = re.sub(r"[()]", " ", text)
        return text

    def custom_tokenizer(text):
        tokens = re.findall(
            r"\d+-\d+\s*\w+"             # 5-8 см
            r"|\d+-[а-яА-ЯїЇєЄіІґҐ]+"    # 1-ша
            r"|[а-яА-ЯїЇєЄіІґҐ]/[а-яА-ЯїЇєЄіІґҐ]+"  # б/трави
            r"|[а-яА-ЯїЇєЄіІґҐ]+",       # звичайні слова
            text.lower()
        )
        stopwords = {"на", "та", "із", "з"}  # можна розширити список
        return [t for t in tokens if t not in stopwords]

    vectorizer = CountVectorizer(
        preprocessor=custom_preprocessor,
        tokenizer=custom_tokenizer
    )

    X_trn_bow = vectorizer.fit_transform(X_trn[operation_column])
    X_val_bow = vectorizer.transform(X_val[operation_column])

    X_trn_bow_df = pd.DataFrame(X_trn_bow.toarray(), 
                                columns=vectorizer.get_feature_names_out(), 
                                index=X_trn.index)
    X_val_bow_df = pd.DataFrame(X_val_bow.toarray(), 
                                columns=vectorizer.get_feature_names_out(), 
                                index=X_val.index)

    print("BoW shape:", X_trn_bow_df.shape)

# %%

if use_tfidf:
    # ============== TF-IDF ==============
    def clean_text(text):
        text = text.lower().strip()
        # прибираємо дужки, крапки, коми, лапки, зайві пробіли
        text = re.sub(r"[()\",.:;!?]", " ", text)
        text = re.sub(r"\s+", " ", text)
        # замінюємо тире на дефіс, бо іноді буває різниця між "–" і "-"
        text = text.replace("–", "-").replace("—", "-")
        return text

    def uk_tokenizer(text):
        # токени: слова, що складаються з літер (укр + англ), цифр і дефісів
        return re.findall(r"[а-щьюяґєіїА-ЩЬЮЯҐЄІЇ0-9\-]+", text)

    tfidf = TfidfVectorizer(
        preprocessor=clean_text,
        tokenizer=uk_tokenizer,
        analyzer="word",
        ngram_range=(1, 3),          # додаємо тріграми для кращого контексту
        max_features=1000,           # більше ознак для багатших представлень
        sublinear_tf=True,
        min_df=2,                    # ігноруємо дуже рідкісні слова
        max_df=0.90,                 # ігноруємо дуже часті слова
        norm="l2"
    )

    X_trn_tfidf = tfidf.fit_transform(X_trn[operation_column])
    X_val_tfidf = tfidf.transform(X_val[operation_column])

    X_trn_tfidf_df = pd.DataFrame(X_trn_tfidf.toarray(), 
                                columns=tfidf.get_feature_names_out(), 
                                index=X_trn.index)
    X_val_tfidf_df = pd.DataFrame(X_val_tfidf.toarray(), 
                                columns=tfidf.get_feature_names_out(), 
                                index=X_val.index)

    print("TF-IDF shape:", X_trn_tfidf_df.shape)

# %%

if use_pca:
    # ============== PCA ==============
    # PCA без фіксації n_components
    pca = PCA(random_state=GLOBAL_RANDOM_SEED)
    pca.fit(trn_embeddings)

    explained_variance = np.cumsum(pca.explained_variance_ratio_)

    # Знаходимо кількість компонент для 95% дисперсії
    n_components_95 = np.argmax(explained_variance >= 0.95) + 1
    print(f"✅ Кількість компонент для 95% дисперсії: {n_components_95}")

    # Побудова графіка
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='.')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% дисперсії')
    plt.axvline(x=n_components_95, color='g', linestyle='--', label=f'{n_components_95} компонент')
    plt.xlabel('Кількість компонент')
    plt.ylabel('Кумулятивна пояснена дисперсія')
    plt.title('Вибір кількості компонент PCA')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Використовуємо PCA зі знайденою кількістю компонент
    pca_final = PCA(n_components=n_components_95, random_state=GLOBAL_RANDOM_SEED)

    X_trn_pca = pca_final.fit_transform(trn_embeddings)
    X_val_pca = pca_final.transform(val_embeddings)

    X_trn_pca_df = pd.DataFrame(X_trn_pca, 
                                columns=[f"pca_{i+1}" for i in range(n_components_95)], 
                                index=X_trn.index)
    X_val_pca_df = pd.DataFrame(X_val_pca, 
                                columns=[f"pca_{i+1}" for i in range(n_components_95)], 
                                index=X_val.index)

    print("PCA shape:", X_trn_pca_df.shape)

# %%

if use_kmeans:
    # ============== KMeans ==============
    # Визначення оптимальної кількості кластерів за допомогою Elbow Method та Silhouette Score
    inertias = []
    silhouette_scores = []
    K_range = range(2, 15)  # перевіримо кластери від 2 до 14

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=GLOBAL_RANDOM_SEED)
        labels = kmeans.fit_predict(trn_embeddings)
        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(trn_embeddings, labels)
        silhouette_scores.append(sil_score)

    # Графік Elbow
    plt.figure(figsize=(8,5))
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Кількість кластерів (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method для KMeans')
    plt.grid(True)
    plt.show()

    # Графік Silhouette Score
    plt.figure(figsize=(8,5))
    plt.plot(K_range, silhouette_scores, 'ro-')
    plt.xlabel('Кількість кластерів (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score для різної кількості кластерів')
    plt.grid(True)
    plt.show()

    # Знайдемо оптимум за silhouette
    best_k = K_range[np.argmax(silhouette_scores)]
    print(f"Оптимальна кількість кластерів за silhouette score: {best_k}")

    selected_k = 8
    # Навчання KMeans з оптимальною кількістю кластерів
    # kmeans_final = KMeans(n_clusters=selected_k, random_state=GLOBAL_RANDOM_SEED)
    kmeans_final = KMeans(n_clusters=best_k, random_state=GLOBAL_RANDOM_SEED)

    X_trn_kmeans = kmeans_final.fit_predict(trn_embeddings)
    X_val_kmeans = kmeans_final.predict(val_embeddings)

    X_trn_kmeans_df = pd.DataFrame(X_trn_kmeans, 
                                columns=['KMeans_operation_cluster'], 
                                index=X_trn.index,
                                )#.astype('category')
    X_val_kmeans_df = pd.DataFrame(X_val_kmeans, 
                                columns=['KMeans_operation_cluster'], 
                                index=X_val.index,
                                )#.astype('category')

    print("KMeans shape:", X_trn_kmeans_df.shape)

# %%

if use_dbscan:
    # ============== DBSCAN ==============
    dbscan = DBSCAN(eps=0.5, min_samples=20)
    X_trn_dbscan = dbscan.fit_predict(trn_embeddings)
    X_val_dbscan = dbscan.predict(val_embeddings)

    X_trn_dbscan_df = pd.DataFrame(X_trn_dbscan, columns=['DBSCAN_operation_cluster'], index=X_trn.index)
    X_val_dbscan_df = pd.DataFrame(X_val_dbscan, columns=['DBSCAN_operation_cluster'], index=X_val.index)

    print("DBSCAN shape:", X_trn_dbscan_df.shape)

# %%

# Вибір одного з методів векторизації
operation_processing_method = [
    # 'bow',
    # 'tfidf',
    # 'pca',
    # 'kmeans',
    # 'dbscan'
    None,
][0]

if operation_processing_method == 'bow':
    X_trn_vectorized_operations = X_trn_bow_df
    X_val_vectorized_operations = X_val_bow_df
elif operation_processing_method == 'tfidf':
    X_trn_vectorized_operations = X_trn_tfidf_df
    X_val_vectorized_operations = X_val_tfidf_df
elif operation_processing_method == 'pca':
    X_trn_vectorized_operations = X_trn_pca_df
    X_val_vectorized_operations = X_val_pca_df
elif operation_processing_method == 'kmeans':
    X_trn_vectorized_operations = X_trn_kmeans_df
    X_val_vectorized_operations = X_val_kmeans_df
# elif operation_processing_method == 'dbscan':
#     X_trn_vectorized_operations = X_trn_dbscan_df
#     X_val_vectorized_operations = X_val_dbscan_df
else:
    X_trn_vectorized_operations = pd.DataFrame(index=X_trn.index)
    X_val_vectorized_operations = pd.DataFrame(index=X_val.index)

if (use_bow | use_tfidf | use_pca | use_kmeans | use_dbscan):
    # Об'єднуємо з основними ознаками
    X_trn_prep = pd.concat([X_trn.drop(columns=[operation_column]), X_trn_vectorized_operations], axis=1)
    X_val_prep = pd.concat([X_val.drop(columns=[operation_column]), X_val_vectorized_operations], axis=1)
else:
    # Для нової версії коду
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
    # "subsample": 0.7,
    # "subsample_freq": 1,
    # "colsample_bytree": 0.6,
    "reg_alpha": 0.2,
    "reg_lambda": 0.8,
    "random_state": GLOBAL_RANDOM_SEED,
    "n_jobs": -1
}

def train_lgbm_with_cv(X, y, params, n_splits=5):
    stratify_col = X['Region_cluster']
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
    mae_mean = []
    mae_lower = []
    mae_upper = []
    r2_mean = []
    mape_mean = []
    mape_lower = []
    mape_upper = []

    def init_and_fit_models(X_train, y_train, cv_mode: bool, X_eval=None, y_eval=None):
        model_mean = LGBMRegressor(objective='regression', **params)
        model_lower = LGBMRegressor(objective='quantile', alpha=0.05, **params)
        model_upper = LGBMRegressor(objective='quantile', alpha=0.95, **params)

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

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, stratify_col), 1):
        X_train, X_eval = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_eval = y.iloc[train_idx], y.iloc[val_idx]

        # print(X_train['Region_cluster'].value_counts(normalize=True))

        model_mean_cv, model_lower_cv, model_upper_cv = init_and_fit_models(X_train, y_train, cv_mode=True, X_eval=X_eval, y_eval=y_eval)

        y_pred_mean = model_mean_cv.predict(X_eval)
        y_pred_lower = model_lower_cv.predict(X_eval)
        y_pred_upper = model_upper_cv.predict(X_eval)

        mae_mean.append(mean_absolute_error(y_eval, y_pred_mean))
        mae_lower.append(mean_absolute_error(y_eval, y_pred_lower))
        mae_upper.append(mean_absolute_error(y_eval, y_pred_upper))

        mape_mean.append(mean_absolute_percentage_error(y_eval, y_pred_mean))
        mape_lower.append(mean_absolute_percentage_error(y_eval, y_pred_lower))
        mape_upper.append(mean_absolute_percentage_error(y_eval, y_pred_upper))

        r2_mean.append(r2_score(y_eval, y_pred_mean))

    model_mean, model_lower, model_upper = init_and_fit_models(X, y, cv_mode=False)

    mean_mae_cv = np.round(np.mean(mae_mean), 4)
    lower_mae_cv = np.round(np.mean(mae_lower), 4)
    upper_mae_cv = np.round(np.mean(mae_upper), 4)

    mean_r2_cv = np.round(np.mean(r2_mean), 4)

    mean_mape_cv = np.round(np.mean(mape_mean), 4)
    lower_mape_cv = np.round(np.mean(mape_lower), 4)
    upper_mape_cv = np.round(np.mean(mape_upper), 4)

    print(f"\nCV MAE on Train: {mean_mae_cv:.4f} ± {np.std(mae_mean):.4f}")
    print(f"CV MAPE on Train: {mean_mape_cv:.4f} ± {np.std(mape_mean):.4f}")

    return {
        'models': {'mean': model_mean, 'lower': model_lower, 'upper': model_upper},
        'cv_mae': {'mean': mean_mae_cv, 'lower': lower_mae_cv, 'upper': upper_mae_cv},
        'cv_r2': {'mean': mean_r2_cv},
        'cv_mape': {'mean': mean_mape_cv, 'lower': lower_mape_cv, 'upper': upper_mape_cv},
    }

start_week = 20
end_week = 28

results_weekly = {}
for w in range(start_week, end_week+1):
    print(w)
    suffix = X_trn_prep.columns.str.extract(r'(\d+)$')[0].astype(float)
    week_cols_to_drop = X_trn_prep.columns[suffix > w]
    X_trn_week = X_trn_prep.drop(columns=week_cols_to_drop)
    results_weekly[w] = train_lgbm_with_cv(X_trn_week, y_trn, lgbm_params, n_splits=5)

# %% 

# Візуалізація точності моделей по тижнях
weeks = results_weekly.keys()
cv_mae_list_for_mean = [v['cv_mae']['mean'] for v in results_weekly.values()]
cv_mae_list_for_lower = [v['cv_mae']['lower'] for v in results_weekly.values()]
cv_mae_list_for_upper = [v['cv_mae']['upper'] for v in results_weekly.values()]
plt.figure(figsize=(10, 6))
plt.plot(weeks, cv_mae_list_for_mean, marker='o', c='blue', label='Mean Model')
plt.plot(weeks, cv_mae_list_for_lower, marker='o', c='green', label='Lower Model')
plt.plot(weeks, cv_mae_list_for_upper, marker='o', c='red', label='Upper Model')
plt.xlabel('Тиждень')
plt.ylabel('CV MAE')
plt.title('Точність моделей по тижнях')
plt.grid(True)
plt.legend()
plt.show()

# %%

# Таблиця з CV метриками
data = []
for w in results_weekly.keys():
    row = {
        'cv_mae_mean': results_weekly[w]['cv_mae']['mean'],
        'cv_mape_mean': results_weekly[w]['cv_mape']['mean'],
        'cv_r2_mean': results_weekly[w]['cv_r2']['mean'],
        'cv_mae_lower': results_weekly[w]['cv_mae']['lower'],
        'cv_mape_lower': results_weekly[w]['cv_mape']['lower'],
        'cv_mae_upper': results_weekly[w]['cv_mae']['upper'],
        'cv_mape_upper': results_weekly[w]['cv_mape']['upper']
    }
    data.append(row)

df_cv_metrics = pd.DataFrame(data, index=results_weekly.keys())
df_cv_metrics

# %%

# Таблиця з метриками на валідаційній вибірці
val_data = []
for w in results_weekly.keys():
    week_cols_to_drop = X_val_prep.columns[X_val_prep.columns.str.extract(r'(\d+)$')[0].astype(float) > w]
    X_val_week = X_val_prep.drop(columns=week_cols_to_drop)
    row = {}
    for model_type in ['mean', 'lower', 'upper']:
        model = results_weekly[w]['models'][model_type]
        y_val_pred = model.predict(X_val_week)
        row[f'val_mae_{model_type}'] = np.round(mean_absolute_error(y_val, y_val_pred), 4)
        row[f'val_mape_{model_type}'] = np.round(mean_absolute_percentage_error(y_val, y_val_pred), 4)
        if model_type == 'mean':
            row[f'val_r2_{model_type}'] = np.round(r2_score(y_val, y_val_pred), 4)
    val_data.append(row)

df_val_metrics = pd.DataFrame(val_data, index=results_weekly.keys())
df_val_metrics

# %% 

# Розрахунок SHAP values для моделей кожного тижня
model_types = [
    'mean', 
    # 'lower', 
    # 'upper'
]

shap_values_weekly = {}
for w in results_weekly.keys():
    shap_values_weekly[w] = {}
    week_cols_to_drop = X_val_prep.columns[X_val_prep.columns.str.extract(r'(\d+)$')[0].astype(float) > w]
    X_val_week = X_val_prep.drop(columns=week_cols_to_drop)
    
    for model_type in model_types:
        model = results_weekly[w]['models'][model_type]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val_week)
        shap_values_weekly[w][model_type] = {
            'shap_values': shap_values,
            'X_val_week': X_val_week
        }

# Побудова графіків SHAP важливості ознак для кожного тижня та моделі
for w in shap_values_weekly.keys():
    for model_type in model_types:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values_weekly[w][model_type]['shap_values'],
            shap_values_weekly[w][model_type]['X_val_week'],
            show=False
        )
        plt.title(f'SHAP важливість ознак: тиждень {w}, модель {model_type}')
        plt.tight_layout()
        plt.show();

# %%