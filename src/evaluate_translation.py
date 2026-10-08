"""Evaluate English->German review translation on a stratified sample of raw reviews.

Outputs:
  outputs/task5_translation_samples.csv  (for manual comparison)
  outputs/task5_translation_metrics.json
No reference German translations exist, so BLEU/ROUGE are computed on the
round-trip (EN -> DE -> EN) against the original and are only a proxy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.translation import back_translate_de_to_en, translate_en_to_de  # noqa: E402

PER_CLASS = 10
SEED = 42


def sentiment(rating: float) -> str:
    return "negative" if rating <= 2 else "neutral" if rating == 3 else "positive"


def main() -> None:
    raw = pd.read_csv(ROOT / "datasets" / "1429_1.csv", usecols=["reviews.text", "reviews.rating"])
    raw = raw.dropna().rename(columns={"reviews.text": "original", "reviews.rating": "rating"})
    raw["sentiment"] = raw["rating"].map(sentiment)
    raw["words"] = raw["original"].str.split().str.len()
    raw = raw[(raw["words"] >= 4) & (raw["words"] <= 120)].drop_duplicates("original")
    sample = pd.concat(
        [g.sample(min(PER_CLASS, len(g)), random_state=SEED) for _, g in raw.groupby("sentiment")]
    ).reset_index(drop=True)

    artifact = joblib.load(ROOT / "models" / "task1_recommended_model.joblib")

    def predict(texts: list[str]) -> list[str]:
        return list(artifact["classifier"].predict(artifact["vectorizer"].transform(texts)))

    results = [translate_en_to_de(t) for t in sample["original"]]
    sample["german"] = [r["translation"] for r in results]
    sample["status"] = [r["status"] for r in results]
    sample["back_translation"] = [back_translate_de_to_en(g) for g in sample["german"]]
    sample["pred_original"] = predict(sample["original"].tolist())
    sample["pred_back"] = predict(sample["back_translation"].tolist())
    sample["sentiment_preserved"] = sample["pred_original"] == sample["pred_back"]

    import sacrebleu
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"])
    ok = sample[sample["status"].str.startswith("translated")]
    bleu = sacrebleu.corpus_bleu(ok["back_translation"].tolist(), [ok["original"].tolist()])
    rouge = [scorer.score(o, b) for o, b in zip(ok["original"], ok["back_translation"])]

    metrics = {
        "n_reviews": int(len(sample)),
        "n_translated": int(len(ok)),
        "round_trip_bleu": round(bleu.score, 2),
        "round_trip_rouge1_f": round(sum(s["rouge1"].fmeasure for s in rouge) / len(rouge), 3),
        "round_trip_rougeL_f": round(sum(s["rougeL"].fmeasure for s in rouge) / len(rouge), 3),
        "sentiment_preserved_rate": round(float(ok["sentiment_preserved"].mean()), 3),
        "sentiment_preserved_by_class": ok.groupby("sentiment")["sentiment_preserved"].mean().round(3).to_dict(),
        "note": "Round-trip proxy metrics; no human German references available.",
    }
    out = ROOT / "outputs"
    cols = ["sentiment", "rating", "original", "german", "back_translation", "pred_original", "pred_back", "sentiment_preserved", "status"]
    sample[cols].to_csv(out / "task5_translation_samples.csv", index=False, encoding="utf-8-sig")
    (out / "task5_translation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
