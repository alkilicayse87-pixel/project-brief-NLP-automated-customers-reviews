from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from scipy.special import softmax


MODEL_PATH = Path(__file__).resolve().parent / "task1_recommended_model.joblib"


def load_model_artifact(path: Path = MODEL_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Sentiment model artifact not found: {path}")
    artifact = joblib.load(path)
    required_keys = {"classifier", "vectorizer", "labels"}
    missing_keys = required_keys.difference(artifact)
    if missing_keys:
        raise ValueError(
            f"Sentiment model artifact is missing required keys: {sorted(missing_keys)}"
        )
    return artifact


class PredictionRequest(BaseModel):
    review: str = Field(min_length=1, max_length=5000)

    @field_validator("review")
    @classmethod
    def strip_and_reject_blank(cls, review: str) -> str:
        review = review.strip()
        if not review:
            raise ValueError("Review must contain non-whitespace text.")
        return review


def create_app(model_path: Path = MODEL_PATH) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.model_artifact = load_model_artifact(model_path)
        yield

    api = FastAPI(
        title="Customer Review Sentiment API",
        description="Predicts negative, neutral, or positive sentiment for one review.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @api.get("/health")
    def health(request: Request) -> dict[str, str]:
        if not hasattr(request.app.state, "model_artifact"):
            raise HTTPException(status_code=503, detail="Sentiment model is not loaded.")
        return {"status": "ok"}

    @api.post("/predict")
    def predict(
        payload: PredictionRequest,
        request: Request,
    ) -> dict[str, Any]:
        model_artifact = request.app.state.model_artifact
        classifier = model_artifact["classifier"]
        vectorizer = model_artifact["vectorizer"]
        labels = model_artifact["labels"]
        features = vectorizer.transform([payload.review])
        scores = np.asarray(classifier.decision_function(features))
        if scores.ndim != 2 or scores.shape[1] != len(labels):
            raise HTTPException(
                status_code=500,
                detail="Model output does not match the configured sentiment classes.",
            )

        relative_scores = softmax(scores, axis=1)[0]
        predicted_label = str(labels[int(np.argmax(relative_scores))])
        return {
            "sentiment": predicted_label,
            "class_scores": {
                str(label): float(score)
                for label, score in zip(labels, relative_scores, strict=True)
            },
            "score_note": (
                "Scores are a softmax of LinearSVC decision values, not calibrated "
                "probabilities."
            ),
        }

    return api


app = create_app()
