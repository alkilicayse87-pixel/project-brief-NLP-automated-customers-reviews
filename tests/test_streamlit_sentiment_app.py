from __future__ import annotations

import os
import unittest

from streamlit.testing.v1 import AppTest


APP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "deploy",
    "streamlit_app.py",
)


class StreamlitSentimentAppTests(unittest.TestCase):
    def test_submitted_review_is_classified_by_saved_model(self) -> None:
        app = AppTest.from_file(APP_PATH, default_timeout=30).run()
        app.text_area[0].set_value("Really good product")
        app.button[0].click().run()

        self.assertFalse(app.exception)
        self.assertTrue(
            any("Predicted sentiment:" in item.value for item in app.subheader)
        )
        self.assertTrue(
            any("not calibrated probabilities" in item.value for item in app.caption)
        )

    def test_blank_review_is_rejected(self) -> None:
        app = AppTest.from_file(APP_PATH, default_timeout=30).run()
        app.text_area[0].set_value("   ")
        app.button[0].click().run()

        self.assertFalse(app.exception)
        self.assertTrue(any("non-empty review" in item.value for item in app.warning))


if __name__ == "__main__":
    unittest.main()
