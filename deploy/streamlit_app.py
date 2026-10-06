from __future__ import annotations

import os

import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


st.set_page_config(page_title="Review Sentiment")
st.title("Customer Review Sentiment")
st.write(
    "Enter a product review to see the sentiment predicted by the Task 1 model."
)


def get_backend_url() -> str | None:
    backend_url = os.getenv("HF_BACKEND_URL")
    if backend_url:
        return backend_url
    try:
        return st.secrets.get("HF_BACKEND_URL")
    except StreamlitSecretNotFoundError:
        return None


backend_url = get_backend_url()
if not backend_url:
    st.error(
        "The Hugging Face backend is not configured. Set HF_BACKEND_URL in "
        "Streamlit Community Cloud app secrets."
    )
    st.stop()

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
        try:
            response = requests.post(
                f"{backend_url.rstrip('/')}/predict",
                json={"review": review},
                timeout=(5, 60),
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            st.error("The Hugging Face backend did not respond before the timeout.")
        except requests.exceptions.HTTPError as exc:
            st.error(
                "The backend rejected the request or returned an error "
                f"(HTTP {exc.response.status_code})."
            )
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not connect to the Hugging Face backend: {exc}")
        else:
            result = response.json()
            st.subheader(f"Predicted sentiment: {result['sentiment'].capitalize()}")
            st.caption(result["score_note"])
            st.bar_chart(result["class_scores"])

st.caption(
    "Reviews are sent to the configured Hugging Face backend for prediction. "
    "The app does not intentionally store submitted text."
)
