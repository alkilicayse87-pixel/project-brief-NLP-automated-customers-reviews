from __future__ import annotations

import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import hstack
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.project_config import DATASET_PATH, MODELS_DIR, OUTPUT_DIR
from src.task3_summarizer import generate_category_articles


def clean_text(text: object) -> str:
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_product_name(name: object) -> str:
    if pd.isna(name):
        return 'Unknown product'
    first_line = str(name).replace('\r', '\n').split('\n', maxsplit=1)[0]
    cleaned = re.sub(r'\s+', ' ', first_line).strip(' ,\t')
    if not cleaned or cleaned.lower() in {'unknown', 'nan', 'none'}:
        return 'Unknown product'
    return cleaned


def load_reviews(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(dataset_path, low_memory=False)
    required = ['reviews.text', 'reviews.rating', 'name']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    df = df.rename(columns={
        'reviews.text': 'review_text',
        'reviews.rating': 'rating',
        'name': 'product_name',
    })
    df['review_text'] = df['review_text'].map(clean_text)
    df['product_name'] = df['product_name'].map(clean_product_name)
    if 'categories' not in df:
        df['categories'] = ''
    df['categories'] = df['categories'].fillna('').astype(str)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating', 'review_text'])
    df = df[df['review_text'].str.len() > 0].copy()
    df['sentiment'] = df['rating'].apply(lambda r: 'negative' if r <= 2 else 'neutral' if r == 3 else 'positive')
    return df


def train_baseline_models(X: pd.Series, y: pd.Series) -> tuple[pd.DataFrame, dict]:
    labels = ['negative', 'neutral', 'positive']
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=42,
        stratify=y_train_val,
    )

    models = {
        'TF-IDF + Logistic Regression': (
            TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english'),
            LogisticRegression(max_iter=5000, random_state=42),
        ),
        'TF-IDF + Logistic Regression (balanced)': (
            TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english'),
            LogisticRegression(class_weight='balanced', max_iter=5000, random_state=42),
        ),
        'TF-IDF + Multinomial Naive Bayes': (
            TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english'),
            MultinomialNB(),
        ),
        'TF-IDF + LinearSVC (balanced)': (
            TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english'),
            LinearSVC(class_weight='balanced', random_state=42, max_iter=5000),
        ),
        'BoW + Logistic Regression (balanced)': (
            CountVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2)),
            LogisticRegression(class_weight='balanced', max_iter=5000, random_state=42),
        ),
    }

    rows = []
    best_validation_macro_f1 = -1.0
    best_validation_accuracy = -1.0
    best_name = ''
    best_vectorizer = None
    best_classifier = None

    for name, (vectorizer, model) in models.items():
        Xtr = vectorizer.fit_transform(X_train)
        Xval = vectorizer.transform(X_validation)
        model.fit(Xtr, y_train)
        validation_predictions = model.predict(Xval)

        accuracy = accuracy_score(y_validation, validation_predictions)
        precision, recall, macro_f1, _ = precision_recall_fscore_support(
            y_validation, validation_predictions, average='macro', zero_division=0
        )
        per_class = precision_recall_fscore_support(
            y_validation,
            validation_predictions,
            labels=labels,
            zero_division=0,
        )

        rows.append({
            'Model': name,
            'Validation accuracy': accuracy,
            'Validation macro precision': precision,
            'Validation macro recall': recall,
            'Validation macro F1': macro_f1,
            'Negative F1': per_class[2][0],
            'Neutral F1': per_class[2][1],
            'Positive F1': per_class[2][2],
            'Validation weighted F1': f1_score(
                y_validation, validation_predictions, average='weighted'
            ),
        })

        if (macro_f1, accuracy) > (
            best_validation_macro_f1,
            best_validation_accuracy,
        ):
            best_validation_macro_f1 = macro_f1
            best_validation_accuracy = accuracy
            best_name = name
            best_vectorizer = vectorizer
            best_classifier = model

    comparison = pd.DataFrame(rows).sort_values(
        ['Validation macro F1', 'Validation accuracy'], ascending=False
    )
    comparison.to_csv(OUTPUT_DIR / 'task1_model_comparison.csv', index=False)

    if best_vectorizer is None or best_classifier is None:
        raise RuntimeError('No sentiment model was successfully selected.')

    # Refit the selected configuration on train + validation, keeping the test set unseen.
    final_vectorizer = clone(best_vectorizer)
    final_classifier = clone(best_classifier)
    X_train_final = pd.concat([X_train, X_validation])
    y_train_final = pd.concat([y_train, y_validation])
    X_train_features = final_vectorizer.fit_transform(X_train_final)
    X_test_features = final_vectorizer.transform(X_test)
    final_classifier.fit(X_train_features, y_train_final)
    test_predictions = final_classifier.predict(X_test_features)

    test_accuracy = accuracy_score(y_test, test_predictions)
    test_precision, test_recall, test_f1, test_support = precision_recall_fscore_support(
        y_test, test_predictions, labels=labels, zero_division=0
    )
    per_class = pd.DataFrame({
        'Model': best_name,
        'Sentiment': labels,
        'Precision': test_precision,
        'Recall': test_recall,
        'F1-score': test_f1,
        'Support': test_support,
    })
    per_class.to_csv(OUTPUT_DIR / 'task1_per_class_metrics.csv', index=False)

    test_metrics = pd.DataFrame([{
        'Model': best_name,
        'Test accuracy': test_accuracy,
        'Test macro precision': precision_recall_fscore_support(
            y_test, test_predictions, average='macro', zero_division=0
        )[0],
        'Test macro recall': precision_recall_fscore_support(
            y_test, test_predictions, average='macro', zero_division=0
        )[1],
        'Test macro F1': precision_recall_fscore_support(
            y_test, test_predictions, average='macro', zero_division=0
        )[2],
        'Test weighted F1': f1_score(y_test, test_predictions, average='weighted'),
        'Test samples': len(y_test),
    }])
    test_metrics.to_csv(OUTPUT_DIR / 'task1_test_metrics.csv', index=False)

    matrix = confusion_matrix(y_test, test_predictions, labels=labels)
    confusion_table = pd.DataFrame(
        matrix,
        index=[f'Actual {label}' for label in labels],
        columns=[f'Predicted {label}' for label in labels],
    )
    confusion_table.to_csv(OUTPUT_DIR / 'task1_confusion_matrix.csv', index=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(f'Test Confusion Matrix — {best_name}')
    ax.set_xlabel('Predicted sentiment')
    ax.set_ylabel('Actual sentiment')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'task1_confusion_matrix.png', dpi=150)
    plt.close(fig)

    artifact = {
        'classifier': final_classifier,
        'vectorizer': final_vectorizer,
        'labels': labels,
        'recommended_model': best_name,
        'selection_metric': 'validation_macro_f1',
        'validation_macro_f1': best_validation_macro_f1,
        'test_accuracy': test_accuracy,
        'test_weighted_f1': float(test_metrics.iloc[0]['Test weighted F1']),
    }
    joblib.dump(artifact, OUTPUT_DIR / 'task1_recommended_model.joblib')
    joblib.dump(artifact, MODELS_DIR / 'task1_recommended_model.joblib')

    return comparison, artifact


def build_cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    product_rows = []
    named_reviews = df.loc[df['product_name'] != 'Unknown product']
    for product_name, group in named_reviews.groupby('product_name', sort=True):
        category_values = sorted(
            {value.strip() for value in group['categories'] if value.strip()}
        )
        review_sample = group.loc[
            group['review_text'].str.len() > 0, 'review_text'
        ].head(20)
        product_rows.append({
            'product_name': product_name,
            'categories_text': ' '.join(category_values),
            'review_document': ' '.join(review_sample),
            'review_count': int(len(group)),
            'mean_rating': float(group['rating'].mean()),
            'negative_share': float((group['sentiment'] == 'negative').mean()),
            'positive_share': float((group['sentiment'] == 'positive').mean()),
        })
    product_summary = pd.DataFrame(product_rows)
    if len(product_summary) <= 5:
        raise ValueError(
            f'Product clustering needs more than five named products; '
            f'found {len(product_summary)}.'
        )

    title_vectorizer = TfidfVectorizer(
        stop_words='english', max_features=1000, ngram_range=(1, 2), sublinear_tf=True
    )
    category_vectorizer = TfidfVectorizer(
        stop_words='english', max_features=500, ngram_range=(1, 2), sublinear_tf=True
    )
    review_vectorizer = TfidfVectorizer(
        stop_words='english', max_features=4000, ngram_range=(1, 2), sublinear_tf=True
    )
    title_features = title_vectorizer.fit_transform(product_summary['product_name']) * 4.0
    category_features = category_vectorizer.fit_transform(
        product_summary['categories_text']
    )
    review_features = review_vectorizer.fit_transform(
        product_summary['review_document']
    ) * 0.25
    features = hstack(
        [title_features, category_features, review_features]
    ).tocsr()

    solution_rows = []
    for n_clusters in (4, 5, 6):
        candidate_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=30)
        candidate_ids = candidate_model.fit_predict(features)
        solution_rows.append({
            'clusters': n_clusters,
            'inertia': candidate_model.inertia_,
            'cosine silhouette': silhouette_score(
                features, candidate_ids, metric='cosine'
            ),
        })
    pd.DataFrame(solution_rows).to_csv(
        OUTPUT_DIR / 'task2_cluster_solution_comparison.csv', index=False
    )

    cluster_model = KMeans(n_clusters=5, random_state=42, n_init=30)
    product_summary['cluster_id'] = cluster_model.fit_predict(features)
    category_names = {
        0: 'Fire Tablets',
        1: 'Kindle E-readers',
        2: 'Echo, Audio & Related Products',
        3: 'Fire TV & Streaming Devices',
        4: 'Fire Kids & Family Tablets',
    }
    cluster_documents = (
        product_summary.groupby('cluster_id')['review_document'].apply(' '.join).sort_index()
    )
    profile_vectorizer = CountVectorizer(
        stop_words='english', max_features=2000, ngram_range=(1, 2)
    )
    profile_features = profile_vectorizer.fit_transform(cluster_documents)
    profile_terms = np.asarray(profile_vectorizer.get_feature_names_out())

    cluster_profiles = []
    for cluster_id, group in product_summary.groupby('cluster_id', sort=True):
        top_products = ' | '.join(
            group.nlargest(5, 'review_count')['product_name'].tolist()
        )
        term_scores = np.asarray(profile_features[cluster_id].todense()).ravel()
        top_terms = ', '.join(profile_terms[term_scores.argsort()[::-1][:10]])
        cluster_profiles.append({
            'cluster_id': int(cluster_id),
            'product_count': int(group.shape[0]),
            'total_reviews': int(group['review_count'].sum()),
            'mean_rating': float(group['mean_rating'].mean()),
            'negative_share': float(group['negative_share'].mean()),
            'positive_share': float(group['positive_share'].mean()),
            'top_products': top_products,
            'cluster_name': category_names[int(cluster_id)],
            'top_terms': top_terms,
        })

    cluster_df = pd.DataFrame(cluster_profiles).sort_values('cluster_id')
    cluster_df.to_csv(OUTPUT_DIR / 'task2_cluster_profiles.csv', index=False)

    product_cluster_map = product_summary[[
        'product_name', 'cluster_id', 'review_count', 'mean_rating',
        'negative_share', 'positive_share', 'categories_text',
    ]].copy()
    product_cluster_map['cluster_name'] = product_cluster_map['cluster_id'].map(
        category_names
    )
    product_cluster_map.to_csv(OUTPUT_DIR / 'task2_product_clusters.csv', index=False)

    review_cluster_map = df[[
        'review_text', 'product_name', 'rating', 'sentiment'
    ]].copy()
    review_cluster_map = review_cluster_map.merge(
        product_cluster_map[['product_name', 'cluster_id', 'cluster_name']],
        on='product_name',
        how='left',
        validate='many_to_one',
    )
    review_cluster_map['cluster_id'] = (
        review_cluster_map['cluster_id'].fillna(-1).astype(int)
    )
    review_cluster_map['cluster_name'] = review_cluster_map['cluster_name'].fillna(
        'Unclassified / missing product name'
    )
    review_cluster_map.to_csv(OUTPUT_DIR / 'task2_review_clusters.csv', index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    counts = (
        review_cluster_map.loc[review_cluster_map['cluster_id'] >= 0]
        .groupby('cluster_name').size().sort_values(ascending=False)
    )
    counts.plot(kind='bar', ax=ax, color=sns.color_palette('Set2', len(counts)))
    ax.set_title('Review count by product meta-category')
    ax.set_xlabel('Product category')
    ax.set_ylabel('Reviews')
    ax.tick_params(axis='x', rotation=35)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'task2_cluster_review_counts.png', dpi=150)
    plt.close(fig)

    return cluster_df


def run_task3_only() -> str:
    """Regenerate the category articles using existing Task 2 product clusters."""
    profile_path = OUTPUT_DIR / 'task2_cluster_profiles.csv'
    product_clusters_path = OUTPUT_DIR / 'task2_product_clusters.csv'
    if not profile_path.is_file() or not product_clusters_path.is_file():
        raise FileNotFoundError(
            'Task 3 only requires task2_cluster_profiles.csv and '
            'task2_product_clusters.csv in the outputs directory. Run the full '
            'pipeline once before using --task3-only.'
        )

    cluster_profiles = pd.read_csv(profile_path)
    product_clusters = pd.read_csv(product_clusters_path)
    profile_columns = {"cluster_id", "cluster_name"}
    missing_profile_columns = profile_columns.difference(cluster_profiles.columns)
    if missing_profile_columns:
        raise ValueError(
            'Task 3 cluster profiles are missing required columns: '
            f'{sorted(missing_profile_columns)}. Re-run Task 2 to refresh them.'
        )
    product_columns = {"product_name", "cluster_id"}
    missing_product_columns = product_columns.difference(product_clusters.columns)
    if missing_product_columns:
        raise ValueError(
            'Task 3 product clusters are missing required columns: '
            f'{sorted(missing_product_columns)}. Re-run Task 2 to refresh them.'
        )
    if product_clusters['product_name'].duplicated().any():
        raise ValueError(
            'Task 3 product clusters contain duplicate product names. '
            'Re-run Task 2 to refresh them.'
        )

    reviews = load_reviews()
    clustered_reviews = reviews.merge(
        product_clusters[['product_name', 'cluster_id']],
        on='product_name',
        how='left',
        validate='many_to_one',
    )
    unknown_ids = set(
        pd.to_numeric(clustered_reviews['cluster_id'], errors='coerce')
        .dropna()
        .astype(int)
    ).difference(set(cluster_profiles['cluster_id'].astype(int)))
    if unknown_ids:
        raise ValueError(
            'Task 3 review assignments refer to cluster IDs missing from the '
            f'saved profiles: {sorted(unknown_ids)}. Re-run Task 2 to refresh them.'
        )

    clustered_reviews['cluster_id'] = (
        pd.to_numeric(clustered_reviews['cluster_id'], errors='coerce')
        .fillna(-1)
        .astype(int)
    )
    category_names = cluster_profiles.set_index('cluster_id')['cluster_name']
    clustered_reviews['cluster_name'] = clustered_reviews['cluster_id'].map(
        category_names
    ).fillna('Unclassified / missing product name')
    clustered_reviews = clustered_reviews[[
        'product_name', 'review_text', 'rating', 'sentiment',
        'cluster_id', 'cluster_name',
    ]]
    clustered_reviews.to_csv(OUTPUT_DIR / 'task2_review_clusters.csv', index=False)
    article, _ = generate_category_articles(cluster_profiles, clustered_reviews)
    return article


def run_all() -> dict[str, pd.DataFrame | str]:
    df = load_reviews()
    sentiment_comparison, artifact = train_baseline_models(df['review_text'], df['sentiment'])
    cluster_df = build_cluster_profiles(df)
    clustered_reviews = pd.read_csv(OUTPUT_DIR / 'task2_review_clusters.csv')
    article, _ = generate_category_articles(cluster_df, clustered_reviews)
    return {
        'sentiment_comparison': sentiment_comparison,
        'clusters': cluster_df,
        'article': article,
        'artifact': artifact,
    }
