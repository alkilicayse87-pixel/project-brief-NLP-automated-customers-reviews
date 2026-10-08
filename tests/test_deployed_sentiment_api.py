from pathlib import Path
import unittest

from fastapi.testclient import TestClient

from deploy.hf_backend.app import create_app


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "outputs" / "task1_recommended_model.joblib"


class SentimentApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client_context = TestClient(create_app(MODEL_PATH))
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)

    def test_health_reports_ready_after_model_load(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_prediction_returns_three_relative_class_scores(self) -> None:
        response = self.client.post(
            "/predict",
            json={"review": "This product is excellent and works perfectly."},
        )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn(result["sentiment"], {"negative", "neutral", "positive"})
        self.assertEqual(
            set(result["class_scores"]),
            {"negative", "neutral", "positive"},
        )
        self.assertAlmostEqual(sum(result["class_scores"].values()), 1.0)
        self.assertIn("not calibrated probabilities", result["score_note"])

    def test_blank_and_oversized_reviews_are_rejected(self) -> None:
        blank = self.client.post("/predict", json={"review": "   "})
        oversized = self.client.post("/predict", json={"review": "x" * 5001})

        self.assertEqual(blank.status_code, 422)
        self.assertEqual(oversized.status_code, 422)


if __name__ == "__main__":
    unittest.main()
