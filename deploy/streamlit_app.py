from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import streamlit as st
from scipy.special import softmax


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "task1_recommended_model.joblib"
)


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


st.set_page_config(page_title="Review Sentiment")
st.title("Customer Review Sentiment")
st.write(
    "Enter a product review to see the sentiment predicted by the Task 1 model."
)

model_artifact = load_model_artifact()

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

st.caption(
    "Predictions run in this app using the saved Task 1 model. "
    "Submitted reviews are not intentionally stored."
)
