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
import torch
import shap
from scipy.sparse import hstack, csr_matrix

from sklearn.metrics import root_mean_squared_error, silhouette_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import KNNImputer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from lightgbm import LGBMRegressor
from lightgbm.callback import early_stopping

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
df_main = df_main.drop(columns=['Year', 'Culture', 'Moisture', 'K', 'WeatherGridId', 'crop', 'County', 'Mex', 'Cluster'])
df_main = df_main[df_main['Area'].notna()]
col_patterns_to_drop = [
                        'CumSum_Precipitation',
                        'CumSum_TempEffective'
                        ]
cols_to_drop = [column for column in df_main.columns
                  if any(substring in column for substring in col_patterns_to_drop)]
df_main = df_main.drop(columns=cols_to_drop)

# initial preparation `operations` dataframe
df_operations.columns = ['operation', 'detailed_operation', 'year_of_start', 'date_of_start', 'total', 'field_id']
df_operations['date_of_start'] = pd.to_datetime(df_operations['date_of_start'])

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

df_operations_merged_rows = df_operations_merged.groupby('field_id', as_index=False)['detailed_operation'].apply(lambda x: ' '.join(x))

# %%

# # ============== Bag of Words ==============
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
bow_features = vectorizer.fit_transform(df_operations_merged_rows['detailed_operation'])

X_bow = pd.DataFrame(
    bow_features.toarray(), 
    columns=vectorizer.get_feature_names_out()
)

vectorized_operations = X_bow

# %%

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
    ngram_range=(1, 2),          # уніграми + біграми
    max_features=4000,
    sublinear_tf=True,
    min_df=1,
    max_df=0.95,
    norm="l2"
)

tfidf_features = tfidf.fit_transform(df_operations_merged_rows['detailed_operation'])   # sparse матриця
X_tfidf = pd.DataFrame(
    tfidf_features.toarray(), 
    columns=tfidf.get_feature_names_out()
)
print("TF-IDF shape:", X_tfidf.shape)

vectorized_operations = X_tfidf

# %%

# ============== Embeddings V1 ==============
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# model_name = "Alibaba-NLP/gte-multilingual-base"

# Завантажуємо токенайзер і модель
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_sentence_embedding(text):
    # Токенізація
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    # Беремо середнє по токенах (mean pooling)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy()

texts = df_operations_merged_rows['detailed_operation'].tolist()

embeddings = np.vstack([get_sentence_embedding(t) for t in texts])
print(embeddings.shape)

# %%

# ============== Embeddings V2 ==============
# model_name_or_path="Alibaba-NLP/gte-multilingual-base"
# model = SentenceTransformer(model_name_or_path, trust_remote_code=True)
# embeddings = model.encode(texts, normalize_embeddings=True)
# print(embeddings.shape)

# sim scores
# scores = model.similarity(embeddings[:1], embeddings[1:])

# %%

# ============== PCA ==============
# PCA без фіксації n_components
pca = PCA(random_state=8)
pca.fit(embeddings)

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
pca_final = PCA(n_components=n_components_95, random_state=8)
emb_pca = pca_final.fit_transform(embeddings)
X_pca = pd.DataFrame(emb_pca, columns=[f"pca_{i+1}" for i in range(n_components_95)])

vectorized_operations = X_pca

# %%

# ============== KMeans ==============
# Визначення оптимальної кількості кластерів за допомогою Elbow Method та Silhouette Score
inertias = []
silhouette_scores = []
K_range = range(2, 15)  # перевіримо кластери від 2 до 14

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=8)
    labels = kmeans.fit_predict(embeddings)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(embeddings, labels)
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
print(f"✅ Оптимальна кількість кластерів за silhouette score: {best_k}")

# Навчання KMeans з оптимальною кількістю кластерів
kmeans = KMeans(n_clusters=best_k, random_state=8)
clusters = kmeans.fit_predict(embeddings)
X_kmeans = pd.DataFrame(clusters, columns=['operation_cluster'])

vectorized_operations = X_kmeans

# %%

# ============== DBSCAN ==============
dbscan = DBSCAN(eps=0.5, min_samples=20)
labels = dbscan.fit_predict(embeddings)

X_dbscan = pd.DataFrame(labels, columns=['operation_dbscan_cluster'])

vectorized_operations = X_dbscan


# %%

# Вибір одного з методів векторизації
vectorized_operations = X_bow
# vectorized_operations = X_tfidf
# vectorized_operations = X_pca
# vectorized_operations = X_kmeans
# vectorized_operations = X_dbscan
# vectorized_operations = pd.DataFrame()

# %%

df_operations_prepared = pd.concat([df_operations_merged_rows, vectorized_operations], axis=1)
df_operations_prepared = df_operations_prepared.drop(columns=['detailed_operation'])

df_main_prepared = (df_main
                    .merge(df_operations_prepared, left_on='Field_id', right_on='field_id')
                    .drop(columns=['Field_id', 'field_id']))

# %%

X = df_main_prepared.drop(columns=['Yield'])
y = df_main_prepared['Yield']

# заповнюємо пропуски, окільки є декілька пропусків в `P`
# imputer = KNNImputer(n_neighbors=5).set_output(transform='pandas')
# X = imputer.fit_transform(X)

# %%

X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=0.15, random_state=8)


# %%
# Model cycle

lgbm_params = {
    "n_estimators": 2000,       # багато дерев, але з early stopping
    "learning_rate": 0.03,      # повільне навчання
    "num_leaves": 31,           # контролює розгалуження (чим менше, тим менш ризикований оверфіт)
    "max_depth": 6,             # глибина дерева
    "min_child_samples": 20,    # мін. розмір листка (запобігає "зубрінню")
    "min_child_weight": 1e-2,   # теж допомагає уникати шуму
    "subsample": 0.7,           # стохастичне вибирання рядків
    "subsample_freq": 1,
    "colsample_bytree": 0.7,    # стохастичне вибирання ознак
    "reg_alpha": 0.2,           # L1-регуляризація
    "reg_lambda": 0.8,          # L2-регуляризація
    "random_state": 8,
    "n_jobs": -1                # використовує всі ядра CPU
}

def train_lgbm_with_cv(X, y, params, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=8)
    scores_mean = []
    scores_lower = []
    scores_upper = []

    def init_and_fit_models(cv_mode=False):
        # Модель для середнього значення
        model_mean = LGBMRegressor(objective='regression', **params)
        # Модель для нижньої межі
        model_lower = LGBMRegressor(objective='quantile', alpha=0.05, **params)
        # Модель для верхньої межі
        model_upper = LGBMRegressor(objective='quantile', alpha=0.95, **params)

        for model in [model_mean, model_lower, model_upper]:
            if cv_mode:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_eval, y_eval)],
                    eval_metric="rmse",
                    callbacks=[early_stopping(stopping_rounds=100, verbose=True)],  # try int in range 20-100
                )
            else:
                model.fit(
                    X, y,
                    eval_metric="rmse",
                )
        return model_mean, model_lower, model_upper

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_train, X_eval = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_eval = y.iloc[train_idx], y.iloc[val_idx]

        model_mean_cv, model_lower_cv, model_upper_cv = init_and_fit_models(cv_mode=True)

        y_pred_mean = model_mean_cv.predict(X_eval)
        y_pred_lower = model_lower_cv.predict(X_eval)
        y_pred_upper = model_upper_cv.predict(X_eval)

        scores_mean.append(root_mean_squared_error(y_eval, y_pred_mean))
        scores_lower.append(root_mean_squared_error(y_eval, y_pred_lower))
        scores_upper.append(root_mean_squared_error(y_eval, y_pred_upper))

    model_mean, model_lower, model_upper = init_and_fit_models(cv_mode=False)

    mean_rmse_cv = np.mean(scores_mean)
    lower_rmse_cv = np.mean(scores_lower)
    upper_rmse_cv = np.mean(scores_upper)
    print(f"\nCV RMSE on Train: {mean_rmse_cv:.4f} ± {np.std(scores_mean):.4f}")

    return {
        'models': {'mean': model_mean, 'lower': model_lower, 'upper': model_upper},
        'cv_rmse': {'mean': mean_rmse_cv, 'lower': lower_rmse_cv, 'upper': upper_rmse_cv},
    }

start_week = 15
end_week = 33

results_weekly = {}
for w in range(start_week, end_week+1):
    print(w)
    suffix = df_main.columns.str.extract(r'(\d+)$')[0].astype(float)
    week_cols_to_drop = df_main.columns[suffix > w]
    X_trn_prep = X_trn.drop(columns=week_cols_to_drop)
    results_weekly[w] = train_lgbm_with_cv(X_trn_prep, y_trn, lgbm_params, n_splits=5)

# %% 

# Візуалізація точності моделей по тижнях
weeks = results_weekly.keys()
cv_rmse_list_for_mean = [v['cv_rmse']['mean'] for v in results_weekly.values()]
cv_rmse_list_for_lower = [v['cv_rmse']['lower'] for v in results_weekly.values()]
cv_rmse_list_for_upper = [v['cv_rmse']['upper'] for v in results_weekly.values()]
plt.figure(figsize=(10, 6))
plt.plot(weeks, cv_rmse_list_for_mean, marker='o', c='blue', label='Mean Model')
plt.plot(weeks, cv_rmse_list_for_lower, marker='o', c='green', label='Lower Model')
plt.plot(weeks, cv_rmse_list_for_upper, marker='o', c='red', label='Upper Model')
plt.xlabel('Тиждень')
plt.ylabel('CV RMSE')
plt.title('Точність моделей по тижнях')
plt.grid(True)
plt.legend()
plt.show()


# %%

# Оцінка моделей на валідаційній вибірці
print("Validation RMSE:")
for w in results_weekly.keys(): 
    print(f"Week {w}:")
    week_cols_to_drop = df_main.columns[df_main.columns.str.extract(r'(\d+)$')[0].astype(float) > w]
    X_val_prep = X_val.drop(columns=week_cols_to_drop)
    # print(X_val_prep.columns)
    
    for model_type, model in results_weekly[w]['models'].items():
        y_val_pred = model.predict(X_val_prep)
        val_rmse = root_mean_squared_error(y_val, y_val_pred)
        print(f"  {model_type.capitalize()} Model RMSE: {val_rmse:.4f}")
    print()


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
    week_cols_to_drop = df_main.columns[df_main.columns.str.extract(r'(\d+)$')[0].astype(float) > w]
    X_val_prep = X_val.drop(columns=week_cols_to_drop)
    
    for model_type in model_types:
        model = results_weekly[w]['models'][model_type]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val_prep)
        shap_values_weekly[w][model_type] = {
            'shap_values': shap_values,
            'X_val_prep': X_val_prep
        }

# Побудова графіків SHAP важливості ознак для кожного тижня та моделі
for w in shap_values_weekly.keys():
    for model_type in model_types:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values_weekly[w][model_type]['shap_values'],
            shap_values_weekly[w][model_type]['X_val_prep'],
            show=False
        )
        plt.title(f'SHAP важливість ознак: тиждень {w}, модель {model_type}')
        plt.tight_layout()
        plt.show();

# %%