from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import pipeline

from src.project_config import OUTPUT_DIR
from src.task3_summarizer import _build_cluster_evidence


DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"


def _clean_review_text(text: str) -> str:
    return " ".join(str(text).split())


def _format_summary(summary: str) -> str:
    if not summary.strip():
        return "The model did not return a usable summary."
    return summary.strip()


def _format_source_examples(examples: list[dict[str, str]]) -> str:
    return " | ".join(
        f"Rating {example['rating']}: {_clean_review_text(example['review'])}"
        for example in examples
    )


def generate_local_category_articles(
    cluster_df: pd.DataFrame,
    reviews: pd.DataFrame,
    *,
    model_name: str = DEFAULT_MODEL,
    output_dir: Path | None = None,
    summarizer: Any | None = None,
    batch_size: int = 2,
) -> tuple[str, pd.DataFrame]:
    """Write evidence-led category articles with a downloaded local BART model.

    Review text is summarized locally. Ratings and product rankings are calculated
    from the supplied data and placed in a transparent article template rather than
    asking the summarization model to invent or infer numeric facts.
    """
    if cluster_df.empty:
        raise ValueError("Task 3 has no cluster profiles to summarize.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if summarizer is None:
        device = 0 if torch.cuda.is_available() else -1
        summarizer = pipeline(
            "summarization",
            model=model_name,
            device=device,
        )

    generated_at = datetime.now(timezone.utc).isoformat()
    all_sections: list[str] = [
        "# Customer Review Guides — Local Transformer\n\n",
        (
            f"> Generated locally with Hugging Face model `{model_name}` "
            f"({generated_at}). Product rankings and statistics are calculated "
            "from the review data; quoted themes are model summaries of sampled "
            "reviews. These are drafts and should be checked against the evidence.\n\n"
        ),
    ]
    article_rows: list[dict[str, Any]] = []

    for _, cluster in cluster_df.iterrows():
        cluster_id = int(cluster["cluster_id"])
        evidence = _build_cluster_evidence(cluster_id, cluster, reviews)

        tasks: list[tuple[str, str, list[dict[str, str]]]] = []
        tasks.append(
            (
                "category_positive",
                "Positive review examples",
                evidence["positive_review_examples"],
            )
        )
        tasks.append(
            (
                "category_negative",
                "Negative review examples",
                evidence["negative_review_examples"],
            )
        )
        for product in evidence["highest_rated_products"]:
            tasks.append(
                (
                    f"product:{product['name']}",
                    product["name"],
                    product["negative_review_examples"],
                )
            )

        lowest = evidence["lowest_rated_product"]
        if lowest is not None and all(
            product["name"] != lowest["name"]
            for product in evidence["highest_rated_products"]
        ):
            tasks.append(
                (
                    f"product:{lowest['name']}",
                    lowest["name"],
                    lowest["negative_review_examples"],
                )
            )

        summarizable = [
            (key, title, examples)
            for key, title, examples in tasks
            if examples
        ]
        summaries: dict[str, str] = {}
        model_summarized: set[str] = set()
        model_inputs: list[tuple[str, str, list[dict[str, str]], str, int]] = []
        for key, title, examples in summarizable:
            text = "\n".join(
                f"Rating {example['rating']}: {_clean_review_text(example['review'])}"
                for example in examples
            )
            input_tokens = len(
                summarizer.tokenizer(
                    text,
                    truncation=True,
                    max_length=1024,
                )["input_ids"]
            )
            if input_tokens < 40:
                summaries[key] = (
                    "Representative review evidence (not model-summarized because "
                    f"the excerpt is short): {_format_source_examples(examples)}"
                )
            else:
                model_inputs.append((key, title, examples, text, input_tokens))

        for start in range(0, len(model_inputs), batch_size):
            batch = model_inputs[start : start + batch_size]
            texts = [text for _, _, _, text, _ in batch]
            shortest_input = min(item[4] for item in batch)
            max_output_tokens = max(20, min(90, int(shortest_input * 0.7)))
            min_output_tokens = min(10, max_output_tokens // 3)
            generated = summarizer(
                texts,
                max_length=max_output_tokens,
                min_length=min_output_tokens,
                do_sample=False,
                truncation=True,
                batch_size=batch_size,
            )
            if len(generated) != len(batch):
                raise RuntimeError(
                    f"Local summarizer returned {len(generated)} summaries for "
                    f"{len(batch)} inputs in cluster {cluster_id}."
                )
            for (key, _, _, _, _), result in zip(batch, generated):
                summary = str(result.get("summary_text", "")).strip()
                if summary.endswith((".", "!", "?")):
                    summaries[key] = _format_summary(summary)
                    model_summarized.add(key)
                else:
                    examples = next(
                        item[2] for item in batch if item[0] == key
                    )
                    summaries[key] = (
                        "Representative review evidence (the model output ended "
                        "mid-sentence, so the source is shown instead): "
                        f"{_format_source_examples(examples)}"
                    )

        lines = [
            f"## {evidence['cluster_name']}\n\n",
            (
                f"This category contains **{evidence['product_count']} products** "
                f"and **{evidence['review_count']:,} reviews**, with an average "
                f"rating of **{evidence['mean_rating']:.2f}/5**. "
                f"{evidence['positive_share']:.1%} of assigned reviews are positive "
                f"and {evidence['negative_share']:.1%} are negative.\n\n"
            ),
            "### Review themes\n\n",
        ]

        positive_summary = summaries.get("category_positive")
        negative_summary = summaries.get("category_negative")
        if positive_summary:
            prefix = (
                "DistilBART summary: "
                if "category_positive" in model_summarized
                else ""
            )
            lines.append(f"- **Positive examples:** {prefix}{positive_summary}\n")
        else:
            lines.append("- **Positive examples:** No positive examples were available.\n")
        if negative_summary:
            prefix = (
                "DistilBART summary: "
                if "category_negative" in model_summarized
                else ""
            )
            lines.append(f"- **Negative examples:** {prefix}{negative_summary}\n\n")
        else:
            lines.append(
                "- **Negative examples:** No negative examples were available "
                "in the selected review sample.\n\n"
            )

        lines.append("### Top 3 products and differences\n\n")
        for rank, product in enumerate(evidence["highest_rated_products"], start=1):
            product_key = f"product:{product['name']}"
            complaint_summary = summaries.get(product_key)
            if complaint_summary:
                prefix = (
                    "DistilBART summary: "
                    if product_key in model_summarized
                    else ""
                )
                complaint_text = prefix + complaint_summary
            elif product["negative_review_examples"]:
                complaint_text = "The model did not return a usable complaint summary."
            else:
                complaint_text = (
                    "No negative review examples were available in the selected "
                    "sample; this is not evidence that the product has no complaints."
                )
            lines.append(
                f"{rank}. **{product['name']}** — "
                f"{product['mean_rating']:.2f}/5 from {product['review_count']:,} "
                f"reviews, including {product['negative_review_count']:,} negative. "
                f"Product-specific complaint evidence: {complaint_text}\n"
            )
        lines.append(
            "\nProducts are ordered by mean rating (with review count as a tie-breaker); "
            "the differences reported here are limited to rating and review evidence.\n\n"
        )

        if lowest is not None:
            worst_key = f"product:{lowest['name']}"
            worst_summary = summaries.get(worst_key)
            if worst_summary:
                prefix = (
                    "DistilBART summary: "
                    if worst_key in model_summarized
                    else ""
                )
                lowest_comment = prefix + worst_summary
            elif lowest["negative_review_examples"]:
                lowest_comment = "The model did not return a usable complaint summary."
            else:
                lowest_comment = (
                    "No negative review examples were available in the selected sample."
                )
            lines.extend(
                [
                    "### Lowest-rated product in this comparison\n\n",
                    (
                        f"**{lowest['name']}** has the lowest average rating among "
                        f"eligible products ({lowest['mean_rating']:.2f}/5 from "
                        f"{lowest['review_count']:,} reviews, including "
                        f"{lowest['negative_review_count']:,} negative). "
                        f"Sampled negative-review evidence: {lowest_comment}\n\n"
                        "A relative ranking alone is not enough evidence to "
                        "recommend avoiding this product.\n\n"
                    ),
                ]
            )

        lines.append(
            "### Buying takeaway\n\n"
            "Use the category average and product-level review counts as context. "
            "Complaint summaries reflect only the sampled reviews processed by the "
            "local model; check the source reviews before making a purchase decision.\n\n"
        )
        article = "".join(lines)
        all_sections.append(article)

        article_rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_name": evidence["cluster_name"],
                "model": model_name,
                "inference": "local",
                "generated_at_utc": generated_at,
                "review_count": evidence["review_count"],
                "product_count": evidence["product_count"],
                "mean_rating": evidence["mean_rating"],
                "negative_share": evidence["negative_share"],
                "positive_share": evidence["positive_share"],
                "article": article,
            }
        )

    article_text = "".join(all_sections)
    destination = OUTPUT_DIR if output_dir is None else Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "task3_local_category_articles.md").write_text(
        article_text,
        encoding="utf-8",
    )
    articles = pd.DataFrame(article_rows)
    articles.to_csv(destination / "task3_local_ai_drafts.csv", index=False)
    return article_text, articles
