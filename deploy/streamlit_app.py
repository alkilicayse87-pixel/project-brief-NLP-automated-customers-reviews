from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import softmax


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "task1_recommended_model.joblib"
INSIGHTS_PATH = ROOT / "outputs" / "dashboard_insights.json"
LOCAL_SUMMARIES_PATH = ROOT / "outputs" / "task3_local_ai_drafts.csv"
SENTIMENTS = ["negative", "neutral", "positive"]
LENGTH_BUCKETS = ["0–49", "50–99", "100–199", "200–399", "400+"]


@st.cache_resource
def load_model_artifact() -> dict[str, Any]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Sentiment model artifact not found: {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)
    required_keys = {"classifier", "vectorizer", "labels"}
    missing_keys = required_keys.difference(artifact)
    if missing_keys:
        raise ValueError(
            f"Sentiment model artifact is missing required keys: {sorted(missing_keys)}"
        )
    return artifact


@st.cache_data
def load_dashboard_data() -> dict[str, Any]:
    if not INSIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"Aggregated dashboard data not found: {INSIGHTS_PATH}. "
            "Generate it locally with `python -m src.build_dashboard_artifacts` "
            "and include outputs/dashboard_insights.json in the deployment."
        )

    with INSIGHTS_PATH.open(encoding="utf-8") as artifact_file:
        insights = json.load(artifact_file)

    required_keys = {
        "schema_version",
        "review_count",
        "category_sentiment",
        "product_metrics",
        "review_length_sentiment",
        "complaint_theme_counts",
    }
    missing_keys = required_keys.difference(insights)
    if missing_keys:
        raise ValueError(
            f"Dashboard data is missing required fields: {sorted(missing_keys)}"
        )
    if insights["schema_version"] != 1:
        raise ValueError(
            f"Unsupported dashboard data schema: {insights['schema_version']}"
        )
    if not insights["review_count"]:
        raise ValueError("Dashboard data contains no reviews.")
    return insights


def render_insights() -> None:
    st.header("Explore review insights")
    st.caption(
        "This dashboard uses aggregate statistics only; source review text is not "
        "included in the deployed insights artifact."
    )

    try:
        insights = load_dashboard_data()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        st.warning(str(exc))
        return

    category_sentiment = pd.DataFrame(insights["category_sentiment"])
    products = pd.DataFrame(insights["product_metrics"])
    length_data = pd.DataFrame(insights["review_length_sentiment"])
    complaint_data = pd.DataFrame(insights["complaint_theme_counts"])
    if category_sentiment.empty or products.empty:
        st.warning("No valid review insights are available to display.")
        return

    categories = sorted(category_sentiment["category"].unique().tolist())
    selected_category = st.selectbox(
        "Product category",
        ["All categories", *categories],
        key="insights_category",
    )
    sentiment_source_label = st.radio(
        "Sentiment shown in charts",
        ["Task 1 model predictions", "Labels derived from star ratings"],
        horizontal=True,
    )
    sentiment_source = (
        "model"
        if sentiment_source_label == "Task 1 model predictions"
        else "rating"
    )

    category_filter = (
        category_sentiment
        if selected_category == "All categories"
        else category_sentiment.loc[
            category_sentiment["category"] == selected_category
        ]
    )
    product_filter = (
        products
        if selected_category == "All categories"
        else products.loc[products["category"] == selected_category]
    )
    selected_counts = category_filter.loc[
        category_filter["source"] == sentiment_source
    ]
    selected_review_count = int(selected_counts["count"].sum())
    selected_product_count = int(product_filter["product_name"].nunique())
    average_rating = (
        float(
            (product_filter["mean_rating"] * product_filter["review_count"]).sum()
            / product_filter["review_count"].sum()
        )
        if not product_filter.empty
        else 0.0
    )
    negative_count = int(
        selected_counts.loc[selected_counts["sentiment"] == "negative", "count"].sum()
    )
    negative_share = (
        negative_count / selected_review_count if selected_review_count else 0.0
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(f"Reviews ({sentiment_source})", f"{selected_review_count:,}")
    kpi2.metric("Products", f"{selected_product_count:,}")
    kpi3.metric("Average rating", f"{average_rating:.2f} / 5")
    kpi4.metric("Negative reviews", f"{negative_share:.1%}")

    left, right = st.columns(2)
    with left:
        st.subheader("Sentiment mix by category")
        chart_data = category_sentiment.loc[
            (category_sentiment["source"] == sentiment_source)
            & (category_sentiment["category"] != "Unclassified / missing product name")
        ]
        category_mix = chart_data.pivot(
            index="category",
            columns="sentiment",
            values="count",
        ).reindex(columns=SENTIMENTS, fill_value=0)
        category_mix = category_mix.div(category_mix.sum(axis=1), axis=0).fillna(0)
        st.bar_chart(category_mix)
        st.caption(f"Share of reviews in each category, by {sentiment_source_label.lower()}.")

    with right:
        st.subheader("Sentiment in current selection")
        sentiment_counts = (
            selected_counts.groupby("sentiment")["count"]
            .sum()
            .reindex(SENTIMENTS, fill_value=0)
            .rename("Reviews")
        )
        st.bar_chart(sentiment_counts)
        st.caption(
            "The saved model predicts sentiment from text. Rating labels are "
            "1–2 stars = negative, 3 = neutral, and 4–5 = positive."
        )

    st.subheader("Review length and sentiment")
    length_by_sentiment = length_data.loc[length_data["source"] == sentiment_source]
    length_mix = length_by_sentiment.pivot(
        index="length_bucket",
        columns="sentiment",
        values="count",
    ).reindex(index=LENGTH_BUCKETS, columns=SENTIMENTS, fill_value=0)
    length_mix = length_mix.div(length_mix.sum(axis=1), axis=0).fillna(0)
    st.bar_chart(length_mix)
    st.caption("Bars show sentiment shares within each review-length range.")

    st.subheader("Products with the highest negative-review share")
    max_reviews = max(1, int(product_filter["review_count"].max()))
    min_support = st.slider(
        "Minimum reviews per product",
        min_value=1,
        max_value=max_reviews,
        value=min(20, max_reviews),
        key="minimum_product_reviews",
    )
    negative_column = f"{sentiment_source}_negative"
    positive_column = f"{sentiment_source}_positive"
    ranked_products = product_filter.loc[
        product_filter["review_count"] >= min_support
    ].copy()
    ranked_products["negative_share"] = (
        ranked_products[negative_column] / ranked_products["review_count"]
    )
    ranked_products["positive_share"] = (
        ranked_products[positive_column] / ranked_products["review_count"]
    )
    ranked_products = ranked_products.sort_values(
        ["negative_share", "review_count"],
        ascending=[False, False],
    ).head(10)
    if ranked_products.empty:
        st.info("No products meet the selected minimum review count.")
    else:
        display_products = ranked_products[
            [
                "product_name",
                "review_count",
                "mean_rating",
                "negative_share",
                "positive_share",
            ]
        ].copy()
        display_products["negative_share"] = display_products[
            "negative_share"
        ].map(lambda value: f"{value:.1%}")
        display_products["positive_share"] = display_products[
            "positive_share"
        ].map(lambda value: f"{value:.1%}")
        display_products["mean_rating"] = display_products["mean_rating"].map(
            lambda value: f"{value:.2f}"
        )
        st.dataframe(display_products, hide_index=True, width="stretch")
        st.caption(
            "Rankings include a support threshold because percentages from small "
            "review counts can be unstable."
        )

    st.subheader("Most-mentioned complaint themes")
    selected_complaints = complaint_data.loc[
        complaint_data["source"] == sentiment_source
    ]
    if selected_category != "All categories":
        selected_complaints = selected_complaints.loc[
            selected_complaints["category"] == selected_category
        ]
    theme_counts = (
        selected_complaints.groupby("theme")["count"]
        .sum()
        .sort_values(ascending=False)
    )
    st.bar_chart(theme_counts)
    st.caption(
        f"Keyword matches among negative reviews according to "
        f"{sentiment_source_label.lower()}. A review can match multiple themes; "
        "these are not model-generated topics."
    )

    product_options = sorted(product_filter["product_name"].unique().tolist())
    chosen_product = st.selectbox(
        "Inspect an individual product",
        product_options,
        key="insights_product",
    )
    selected_product = product_filter.loc[
        product_filter["product_name"] == chosen_product
    ].iloc[0]
    st.write(
        f"**{chosen_product}** — {int(selected_product['review_count']):,} reviews, "
        f"average rating {selected_product['mean_rating']:.2f} / 5"
    )
    st.bar_chart(
        pd.Series(
            {
                sentiment: int(selected_product[f"{sentiment_source}_{sentiment}"])
                for sentiment in SENTIMENTS
            },
            name="Reviews",
        )
    )

    download = ranked_products.drop(
        columns=[
            f"{source}_{sentiment}"
            for source in ("model", "rating")
            for sentiment in SENTIMENTS
        ],
        errors="ignore",
    ).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download ranked product insights (CSV)",
        data=download,
        file_name="customer_product_insights.csv",
        mime="text/csv",
    )

    if LOCAL_SUMMARIES_PATH.is_file():
        summaries = pd.read_csv(LOCAL_SUMMARIES_PATH, low_memory=False)
        if {"cluster_name", "article"}.issubset(summaries.columns):
            st.subheader("Local AI category guides")
            if selected_category == "All categories":
                visible_summaries = summaries
            else:
                visible_summaries = summaries.loc[
                    summaries["cluster_name"] == selected_category
                ]
            for row in visible_summaries.itertuples(index=False):
                with st.expander(str(row.cluster_name)):
                    st.markdown(str(row.article))

    st.caption(
        "The saved review-cluster data does not include review dates, so this "
        "dashboard does not show a time trend."
    )


def render_single_review_predictor(model_artifact: dict[str, Any]) -> None:
    st.header("Analyze one review")
    st.write(
        "Enter a product review to see the sentiment predicted by the Task 1 model."
    )

    with st.form("sentiment_form"):
        review = st.text_area(
            "Customer review",
            max_chars=5000,
            height=180,
            placeholder="Paste a product review here...",
        )
        submitted = st.form_submit_button("Analyze review")

    if submitted:
        if not review.strip():
            st.warning("Enter a non-empty review.")
        else:
            classifier = model_artifact["classifier"]
            vectorizer = model_artifact["vectorizer"]
            labels = model_artifact["labels"]
            features = vectorizer.transform([review.strip()])
            scores = np.asarray(classifier.decision_function(features))
            if scores.ndim != 2 or scores.shape[1] != len(labels):
                raise ValueError(
                    "Model output does not match the configured sentiment classes."
                )

            relative_scores = softmax(scores, axis=1)[0]
            predicted_label = str(labels[int(np.argmax(relative_scores))])
            class_scores = {
                str(label): float(score)
                for label, score in zip(labels, relative_scores, strict=True)
            }
            st.subheader(f"Predicted sentiment: {predicted_label.capitalize()}")
            st.caption(
                "Scores are a softmax of LinearSVC decision values, not calibrated "
                "probabilities."
            )
            st.bar_chart(class_scores)


st.set_page_config(page_title="Customer Review Insights", layout="wide")
st.title("Customer Review Insights")
st.write(
    "Explore sentiment, product ratings, review length, and complaint themes, "
    "or try the saved sentiment classifier on a review."
)

try:
    model_artifact = load_model_artifact()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

dashboard_tab, predictor_tab = st.tabs(
    ["Review insights dashboard", "Single-review predictor"]
)
with dashboard_tab:
    render_insights()
with predictor_tab:
    render_single_review_predictor(model_artifact)

st.caption(
    "Predictions run in this app using the saved Task 1 model. "
    "Submitted reviews are not intentionally stored."
)
