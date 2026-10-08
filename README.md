# NLP Automated Customer Reviews

This project builds an NLP-based review intelligence pipeline for Amazon customer feedback. It combines sentiment classification, product clustering, and AI-style category summaries to turn a noisy review dataset into actionable product insights.

## Project Goal

The goal is to help a business understand customer perception at scale by automatically:
- classifying reviews as positive, neutral, or negative,
- grouping related products into broader meta-categories,
- generating short buyer-focused summaries for each cluster.

This is useful for product teams, e-commerce teams, and marketing teams that need to quickly identify strengths, recurring complaints, and category trends without reading thousands of reviews manually.

## Business Problem

Customer review data is often large, inconsistent, and fragmented across many products. Manual analysis is slow and unreliable. This project addresses that problem with NLP methods that help extract meaningful patterns from real-world review text.

## Dataset

The project uses the Amazon review dataset stored in `datasets/1429_1.csv`, with columns such as:
- product name
- product category metadata
- review text
- star rating

The dataset includes realistic data-quality issues, such as missing values, duplicated review records, and inconsistent product naming, which makes it a good representation of real-world catalog data.

## Main Tasks Completed

### Task 1: Sentiment Analysis

A supervised NLP pipeline was built to classify review text into:
- negative
- neutral
- positive

Five TF-IDF / Bag-of-Words classifier configurations are compared. The data is split stratified into 60% training, 20% validation, and 20% test. Candidate models are selected using **validation macro F1** (to account for class imbalance); the chosen configuration is then refit on the combined training and validation data. The test set is used only once for final evaluation.

The selected model is TF-IDF + class-balanced LinearSVC. On the untouched test set (6,925 reviews):
- Accuracy: **90.09%**
- Macro F1: **51.17%**
- Weighted F1: **90.59%**

Per-class test metrics:

| Sentiment | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negative | 35.56% | 39.51% | 37.43% | 162 |
| Neutral | 18.78% | 23.67% | 20.94% | 300 |
| Positive | 95.87% | 94.45% | 95.15% | 6,463 |

The dataset is strongly imbalanced toward positive reviews. The model performs much better on positive reviews than on the smaller negative and neutral classes, so accuracy alone is not sufficient to describe performance.

Test confusion matrix (rows are actual labels, columns are predicted labels):

| Actual \ Predicted | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Negative | 64 | 34 | 64 |
| Neutral | 30 | 71 | 199 |
| Positive | 86 | 273 | 6,104 |

![Task 1 test confusion matrix](outputs/task1_confusion_matrix.png)

The confusion matrix shows that many negative and neutral examples are classified as positive, despite strong performance on the majority positive class.

Task 1 evaluation outputs:
- `outputs/task1_model_comparison.csv` — validation results for all five candidates
- `outputs/task1_test_metrics.csv` — final test-set metrics for the selected model
- `outputs/task1_per_class_metrics.csv` — precision, recall, F1, and support for each test class
- `outputs/task1_confusion_matrix.csv` — confusion-matrix counts
- `outputs/task1_confusion_matrix.png` — visual confusion matrix
- `outputs/task1_recommended_model.joblib` — selected model refit on train + validation

The complete, step-by-step, runnable walkthrough—with markdown explanations, Python cells, charts, model comparisons, and saved artifacts—is in [`notebooks/Task 1.ipynb`](notebooks/Task%201.ipynb). Open it in VS Code's Explorer under the `notebooks` folder and run cells from top to bottom.

### Task 2: Product Category Clustering

The product-level K-Means workflow combines TF-IDF features from cleaned product names, catalog category metadata, and sampled review text. It compares 4-, 5-, and 6-cluster solutions, then uses five clusters with a fixed random seed. Unknown product names remain unclassified instead of being treated as a product category. Cluster contents are inspected before applying human-readable labels; labels and cluster boundaries should be reviewed if the dataset or settings change.

This dataset produced **40 named-product records** for clustering. The resulting five categories are **Fire Tablets**, **Kindle E-readers**, **Echo, Audio & Related Products**, **Fire TV & Streaming Devices**, and **Fire Kids & Family Tablets**. The cosine silhouette scores for 4/5/6 clusters are approximately 0.105 / 0.111 / 0.130, indicating substantial overlap. Although six clusters score slightly higher, inspection showed that this split isolates a very small legacy-product/accessory group (39 reviews); five was retained as a more useful broad-category compromise. The result is exploratory, not a claim of sharply separated groups.

The complete runnable walkthrough—with preprocessing, feature construction, 4–6 cluster comparison, representative-product inspection, visualizations, category naming, and output generation—is in [`notebooks/Task 2.ipynb`](notebooks/Task%202.ipynb). Open it in VS Code Explorer under `notebooks` and run the cells from top to bottom.

Task 2 outputs:
- `outputs/task2_cluster_solution_comparison.csv` — inertia and silhouette scores for 4, 5, and 6 clusters
- `outputs/task2_cluster_profiles.csv`
- `outputs/task2_product_clusters.csv`
- `outputs/task2_review_clusters.csv`
- `outputs/task2_cluster_review_counts.png`

### Task 3: LLM-Generated Category Summaries

Task 3 includes a local pretrained-transformer route and an optional hosted API route. The recommended local model is Hugging Face `sshleifer/distilbart-cnn-12-6` (DistilBART CNN/DailyMail), run with Transformers and PyTorch. It downloads weights on first use (about 1.2 GB) and performs inference on the local CPU or CUDA device without an API key. For each Task 2 product cluster, the program builds review/rating evidence, selects the three highest-rated eligible products and lowest-rated eligible product, and includes sampled negative excerpts tied to each product when available. DistilBART summarizes review excerpts; the Python code calculates and formats rankings and statistics from the data. Unknown-name products stay unclassified. Because abstractive summaries may distort or add details, treat articles as drafts and verify them against the reviews before publication.

The local walkthrough and runnable generation cell are in [`notebooks/Task 3.ipynb`](notebooks/Task%203.ipynb); choose the project `.venv-5` kernel in VS Code and run top to bottom. If needed, install dependencies from a PowerShell terminal with `.\.venv-5\Scripts\python.exe -m pip install -r requirements.txt`. The local implementation is `src/task3_local_summarizer.py`, and it writes distinct outputs so it does not overwrite hosted-API drafts. DistilBART was run locally on all five categories in this workspace; it ran on CPU with no OpenAI key. The saved local outputs are the actual model-run artifacts, not mock responses. Where the model returns a visibly incomplete sentence, the article uses the sampled source reviews instead of presenting a cut-off phrase as a summary.

An optional OpenAI Chat Completions API route still uses `gpt-4o-mini` by default. It has three selectable prompt variants and an optional live comparison experiment. Review excerpts are sent to OpenAI and API usage may incur charges. The API key must be supplied through `OPENAI_API_KEY` or notebook `getpass`; never hardcode or commit it.

The notebook also includes the OpenAI API route with three prompt variants, a human evaluation rubric, safe key setup, optional live generation, and output checks. Its mock test validates product-specific evidence and distinct prompt instructions without API calls. **The local DistilBART deliverable is complete and has been run for all five categories.** The optional live OpenAI prompt comparison has not been run; do not claim a live result or winning prompt. That optional comparison makes three API requests, sends sampled review excerpts to OpenAI, and may incur charges.

The key is read from `OPENAI_API_KEY`, never from a hard-coded source value. You can also set it in Windows PowerShell for the current terminal session:

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:OPENAI_MODEL = "gpt-4o-mini"  # Optional; this is already the default
python run_project.py --task3-only
```

Full hosted-API Task 3 generation makes one API request per cluster (five with the current Task 2 outputs). The optional prompt experiment makes three additional requests using the same evidence for one representative category; compare outputs manually with the notebook rubric, then select a prompt variant for full generation. Review excerpts are sent to OpenAI as part of those requests, and API use may incur charges or be subject to rate limits. The run fails explicitly if the key is missing or a request fails; it does not substitute a template and label it AI-generated. The API output CSV records the model, prompt variant, generation time, source evidence statistics, response ID when returned, and token usage when reported. The existing API-named Task 3 files (`task3_category_articles.md` and `task3_ai_drafts.csv`) are legacy drafts and are **not verified as live API results**. The separate `task3_local_*` files are the verified local DistilBART run.

To rerun just Task 3 without retraining sentiment models or reclustering, use the existing Task 2 outputs:

```powershell
python run_project.py --task3-only
```

Generated files:
- `outputs/task3_local_category_articles.md` — locally generated DistilBART category guides
- `outputs/task3_local_ai_drafts.csv` — local model name, inference provenance, evidence metadata, and article per category
- `outputs/task3_category_articles.md` — combined category guides
- `outputs/task3_ai_drafts.csv` — one generated article and provenance/evidence metadata per cluster
- `outputs/task3_prompt_experiment.csv` — optional live three-prompt comparison; created only if you run the paid OpenAI experiment
- `outputs/task2_review_clusters.csv` — cluster-assigned review evidence used to construct prompts

## Task 4: Public Sentiment Demo and Review Insights Dashboard

**Live demo:** [Open the Streamlit app](https://project-brief-nlp-automated-customers-reviews-4z3nd6dqfyj7ba8x.streamlit.app/). Positive, neutral, negative, blank, and 5,000-character review checks passed.

The Streamlit app loads the fitted Task 1 vectorizer and classifier from `models/task1_recommended_model.joblib` and offers both single-review predictions and an interactive insights dashboard. The dashboard applies the saved classifier to the saved reviews in batches, caches the predictions, and lets viewers switch between model predictions and labels derived from star ratings. It filters by product category and includes sentiment shares by category, review-length comparisons, product rating and negative-share rankings, keyword-based complaint themes, individual product sentiment, and CSV export. If the local DistilBART Task 3 output is present, category guides are also shown.

The dashboard reads the compact, aggregate-only `outputs/dashboard_insights.json` artifact alongside the model. Generate it locally after running the project pipeline with `python -m src.build_dashboard_artifacts`, then include that JSON file in the deployment. The original review text is not included in the public dashboard artifact. Complaint themes are transparent keyword matches against reviews classified as negative by the selected sentiment source, not topics predicted by a trained model; a review may match multiple themes. Aggregate model predictions are descriptive and are not held-out test metrics; use the Task 1 test evaluation for model performance. The saved review-cluster data does not include review dates, so the dashboard does not show sentiment over time. Product rankings use a minimum-review threshold because percentages from very small samples can be misleading.

Single-review predictions run in the app without a separate API host or API secrets. The displayed class scores are softmax-normalized LinearSVC decision values, not calibrated probabilities. Submitted reviews are not intentionally stored. The app and its pinned model dependencies are in `deploy/`; the walkthrough, code references, local checks, and publishing steps are in [`notebooks/Task 4.ipynb`](notebooks/Task%204.ipynb).

The Task 1 model artifact is included in this public repository for Streamlit Community Cloud to load. Do not commit private review data, access tokens, or `.streamlit/secrets.toml`.

### Deployment details

The app is deployed from `alkilicayse87-pixel/project-brief-NLP-automated-customers-reviews`, branch `main`, with `deploy/streamlit_app.py` as its entry point. Streamlit Community Cloud installs pinned model dependencies from `deploy/requirements.txt`.

For local development, install `deploy/requirements.txt`, run `python -m src.build_dashboard_artifacts` after the analysis outputs exist, then start `streamlit run deploy/streamlit_app.py`. The public app needs `models/task1_recommended_model.joblib` and `outputs/dashboard_insights.json`; it does not need the source review CSV, a backend URL, or secrets.

## Task 5: English → German Review Translation

`src/translation.py` translates English reviews to German with the Hugging Face `Helsinki-NLP/opus-mt-en-de` model. It normalizes informal text and abbreviations first (e.g. "w/o", "thx"), keeps product names (Kindle, Fire, Echo, Alexa, ...) exactly as written, passes German reviews through unchanged, and flags other languages as unsupported. The app has a "Translate to German" tab (requires transformers, torch, sentencepiece, sacremoses, langdetect).

`src/evaluate_translation.py` translates 30 sampled reviews (10 per sentiment) and writes `outputs/task5_translation_samples.csv` for manual comparison plus `outputs/task5_translation_metrics.json`. No human German references exist, so BLEU/ROUGE are round-trip (EN→DE→EN) proxies: BLEU ≈ 55, ROUGE-1 ≈ 0.83, and the Task 1 sentiment prediction was preserved for 80% of reviews (positive 100%, negative/neutral 70%).

Observed issues: wrong-sense words ("Tablette" = pill, and "Great tablet" read as the name "Große"; both fixed with pre/post-edits, including neuter gender for "Tablet"), literal word-for-word errors in misspelled reviews ("as i though" → "as me"), and mild negatives drifting toward neutral after back-translation.

## Repository Structure

```text
.
├── deploy/
│   ├── hf_backend/
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── publish_hf_space.py
│   ├── streamlit_app.py
│   └── requirements.txt
├── app/
│   └── app.py
├── datasets/
│   └── 1429_1.csv
├── models/
│   └── task1_recommended_model.joblib
├── notebooks/
│   ├── Task 1.ipynb
│   ├── Task 2.ipynb
│   ├── Task 3.ipynb
│   └── Task 4.ipynb
├── outputs/
│   ├── task1_model_comparison.csv
│   ├── task1_per_class_metrics.csv
│   ├── task1_test_metrics.csv
│   ├── task1_confusion_matrix.csv
│   ├── task1_confusion_matrix.png
│   ├── task1_recommended_model.joblib
│   ├── task2_cluster_profiles.csv
│   ├── task2_cluster_solution_comparison.csv
│   ├── task2_cluster_review_counts.png
│   ├── task2_product_clusters.csv
│   ├── task2_review_clusters.csv
│   ├── task3_ai_drafts.csv
│   ├── task3_local_ai_drafts.csv
│   ├── task3_local_category_articles.md
│   └── task3_category_articles.md
├── src/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── project_config.py
│   ├── task3_local_summarizer.py
│   └── task3_summarizer.py
├── tests/
│   └── test_deployed_sentiment_api.py
├── .gitignore
├── README.md
├── requirements.txt
├── run_project.py
└── ...
```

## Reproduction

To reproduce the full analysis pipeline, set `OPENAI_API_KEY` first if you want the optional hosted Task 3 API generation (which may incur charges), then run:

```bash
python run_project.py
```

## Requirements

```bash
pip install -r requirements.txt
```

## Notes

- The dataset files are intentionally excluded from version control because they are large.
- The project focuses on practical NLP workflows using a real-world review dataset.
- The clustering and summary layers are useful for exploratory product intelligence and can be expanded with more advanced LLM-based summarization in later iterations.

## Summary

This project demonstrates a complete NLP workflow for customer review analysis, from raw dataset preparation to model training, clustering, and summary generation. It is structured to be easy to run, extend, and present in a portfolio or academic project setting.
