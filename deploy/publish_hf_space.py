from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, LocalTokenNotFoundError


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parent / "hf_backend"
MODEL_PATH = ROOT / "outputs" / "task1_recommended_model.joblib"


def main() -> None:
    token = os.getenv("HF_TOKEN")
    space_id = os.getenv("HF_SPACE_ID")
    if not space_id or len(space_id.split("/")) != 2:
        raise RuntimeError(
            "Set HF_SPACE_ID to your Hugging Face namespace and Space name, "
            "for example your-username/customer-review-sentiment-api."
        )
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Task 1 model artifact is missing: {MODEL_PATH}. Train Task 1 first."
        )

    api = HfApi(token=token)
    try:
        api.whoami()
    except (HfHubHTTPError, LocalTokenNotFoundError) as exc:
        raise RuntimeError(
            "Hugging Face authentication is unavailable. Run `hf auth login` "
            "with a write-enabled token or set HF_TOKEN locally."
        ) from exc
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True,
    )
    api.update_repo_settings(
        repo_id=space_id,
        repo_type="space",
        private=False,
    )

    with tempfile.TemporaryDirectory(prefix="sentiment-hf-space-") as temp_dir:
        staging_dir = Path(temp_dir)
        for filename in ("app.py", "Dockerfile", "requirements.txt"):
            shutil.copy2(BACKEND_DIR / filename, staging_dir / filename)
        shutil.copy2(MODEL_PATH, staging_dir / "task1_recommended_model.joblib")
        api.upload_folder(
            repo_id=space_id,
            repo_type="space",
            folder_path=staging_dir,
            commit_message="Deploy customer review sentiment API",
        )

    username, space_name = space_id.split("/", maxsplit=1)
    print(f"Space page: https://huggingface.co/spaces/{space_id}")
    print(f"Backend URL: https://{username}-{space_name}.hf.space")


if __name__ == "__main__":
    main()
