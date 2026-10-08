from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
REVIEW_DATA_PATH = OUTPUT_DIR / "task2_review_clusters.csv"
MODEL_PATH = ROOT / "models" / "task1_recommended_model.joblib"
INSIGHTS_PATH = OUTPUT_DIR / "dashboard_insights.json"
SENTIMENTS = ("negative", "neutral", "positive")
SENTIMENT_SOURCES = {
    "model": "model_sentiment",
    "rating": "sentiment",
}
COMPLAINT_THEMES = {
    "Battery & charging": r"\b(?:battery|charge|charging|charger)\b",
    "Connectivity": r"\b(?:wifi|wi fi|bluetooth|connect|connection|internet)\b",
    "Performance": r"\b(?:slow|freeze|freezes|frozen|lag|crash|crashes|restart)\b",
    "Ads & apps": r"\b(?:ad|ads|advert|advertisement|app|apps|application|store)\b",
    "Screen & display": r"\b(?:screen|display|resolution|pixel|pixels)\b",
    "Price & value": r"\b(?:price|priced|expensive|overpriced|cost|value)\b",
    "Durability": r"\b(?:broke|broken|defect|defective|stopped working|damage)\b",
    "Delivery & accessories": (
        r"\b(?:delivery|shipping|arrived|package|packaging|cable|cover)\b"
    ),
}
LENGTH_BINS = (-1, 49, 99, 199, 399, float("inf"))
LENGTH_LABELS = ("0–49", "50–99", "100–199", "200–399", "400+")
BATCH_SIZE = 1024


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found: {path}. Run the project pipeline first."
        )


def _predict_sentiments(reviews: pd.DataFrame, artifact: dict[str, Any]) -> list[str]:
    classifier = artifact["classifier"]
    vectorizer = artifact["vectorizer"]
    predictions: list[str] = []
    for start in range(0, len(reviews), BATCH_SIZE):
        batch = reviews["review_text"].iloc[start : start + BATCH_SIZE]
        features = vectorizer.transform(batch)
        predictions.extend(str(value) for value in classifier.predict(features))
    return predictions


def build_dashboard_artifact(
    review_data_path: Path = REVIEW_DATA_PATH,
    model_path: Path = MODEL_PATH,
    output_path: Path = INSIGHTS_PATH,
) -> Path:
    _require_file(review_data_path, "Task 2 review-cluster data")
    _require_file(model_path, "Task 1 sentiment model")

    reviews = pd.read_csv(review_data_path, low_memory=False)
    required_columns = {
        "review_text",
        "product_name",
        "rating",
        "sentiment",
        "cluster_name",
    }
    missing_columns = required_columns.difference(reviews.columns)
    if missing_columns:
        raise ValueError(
            f"Review-cluster data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    reviews["review_text"] = reviews["review_text"].fillna("").astype(str)
    reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce")
    reviews["sentiment"] = reviews["sentiment"].astype(str).str.lower()
    reviews["review_length"] = reviews["review_text"].str.len()
    reviews = reviews.loc[
        reviews["rating"].notna() & reviews["sentiment"].isin(SENTIMENTS)
    ].copy()
    reviews["model_sentiment"] = _predict_sentiments(
        reviews,
        joblib.load(model_path),
    )
    reviews["length_bucket"] = pd.cut(
        reviews["review_length"],
        bins=LENGTH_BINS,
        labels=LENGTH_LABELS,
    ).astype(str)

    category_sentiment: list[dict[str, Any]] = []
    for category, group in reviews.groupby("cluster_name", dropna=False, sort=True):
        category_name = str(category) if pd.notna(category) else "Unclassified"
        for source, column in SENTIMENT_SOURCES.items():
            counts = group[column].value_counts()
            for sentiment in SENTIMENTS:
                category_sentiment.append(
                    {
                        "category": category_name,
                        "source": source,
                        "sentiment": sentiment,
                        "count": int(counts.get(sentiment, 0)),
                    }
                )

    product_metrics: list[dict[str, Any]] = []
    for (category, product), group in reviews.groupby(
        ["cluster_name", "product_name"], dropna=False, sort=True
    ):
        record: dict[str, Any] = {
            "category": str(category) if pd.notna(category) else "Unclassified",
            "product_name": str(product),
            "review_count": int(len(group)),
            "mean_rating": float(group["rating"].mean()),
        }
        for source, column in SENTIMENT_SOURCES.items():
            counts = group[column].value_counts()
            for sentiment in SENTIMENTS:
                record[f"{source}_{sentiment}"] = int(counts.get(sentiment, 0))
        product_metrics.append(record)

    review_length_sentiment: list[dict[str, Any]] = []
    for source, column in SENTIMENT_SOURCES.items():
        length_counts = (
            reviews.groupby(["length_bucket", column], observed=False)
            .size()
            .rename("count")
            .reset_index()
        )
        for row in length_counts.itertuples(index=False):
            review_length_sentiment.append(
                {
                    "length_bucket": str(row[0]),
                    "sentiment": str(row[1]),
                    "count": int(row[2]),
                    "source": source,
                }
            )

    complaint_theme_counts: list[dict[str, Any]] = []
    for source, sentiment_column in SENTIMENT_SOURCES.items():
        negative = reviews.loc[reviews[sentiment_column] == "negative"]
        for category, category_reviews in negative.groupby(
            "cluster_name", dropna=False, sort=True
        ):
            for theme, pattern in COMPLAINT_THEMES.items():
                count = int(
                    category_reviews["review_text"]
                    .str.contains(pattern, case=False, regex=True, na=False)
                    .sum()
                )
                complaint_theme_counts.append(
                    {
                        "category": (
                            str(category) if pd.notna(category) else "Unclassified"
                        ),
                        "source": source,
                        "theme": theme,
                        "count": count,
                    }
                )

    artifact = {
        "schema_version": 1,
        "sentiment_sources": {
            "model": "Task 1 LinearSVC predictions",
            "rating": "Sentiment labels derived from star ratings",
        },
        "review_count": int(len(reviews)),
        "category_sentiment": category_sentiment,
        "product_metrics": product_metrics,
        "review_length_sentiment": review_length_sentiment,
        "complaint_theme_counts": complaint_theme_counts,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as artifact_file:
        json.dump(artifact, artifact_file, ensure_ascii=False, separators=(",", ":"))
    return output_path


if __name__ == "__main__":
    print(f"Created privacy-preserving dashboard data: {build_dashboard_artifact()}")
