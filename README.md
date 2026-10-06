![logo_ironhack_blue](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Project | NLP Automated Customer Reviews

<br>

## Project Goal

This project aims to develop a product review system powered by NLP models that aggregate customer feedback from different sources. The key tasks include classifying reviews, clustering product categories, and using generative AI to summarize reviews into recommendation articles.

<br>

## Problem Statement

With thousands of reviews available across multiple platforms, manually analyzing them is inefficient. This project seeks to automate the process using NLP models to extract insights and provide users with valuable product recommendations.

<br>

## Datasets

- **Primary Dataset**: [Amazon Product Reviews from Kaggel](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products/data)
   - This dataset contains three CSV files, with significant overlap between them (many reviews appear in multiple files).
   - The file `1429_1.csv` contains over 34,000 samples and is sufficient for completing this project.
   - You may also choose to use the other files, but doing so will require additional cleaning and deduplication.
   <!--
   Note on the id & name fields:

   - Some name values are corrupted — two unrelated product names are concatenated together in the same field (this is the type of data quality issue you may encounter when working with real-world datasets).
   - In several cases, the same id is attached to genuinely different products (e.g. one id covered an Echo, a Fire Tablet, a Kindle cover, a USB charger, and even "Coconut Water Red Tea")
   - This affected 21 of 89 product IDs (~11% of all reviews)

   Ideal fix: clean the name field (kept only the first product name segment) and treated id as unreliable going forward — using the cleaned product name as the trusted identifier for grouping and analysis instead.
   -->

- **Larger Dataset**: [Amazon Reviews Dataset](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews)

- **Additional Datasets**: You are free to use other datasets from sources like HuggingFace, Kaggle, or any other platform.


Notes:
- In this project you'll be working with realistic, real-world datasets.
- Spend some time exploring and understanding the dataset. You may need to fix data quality issues, discard irrelevant features, handle missing values, and make other preprocessing decisions before training your model.
- Add your `datasets` folder (or at least the CSV files in it) to `.gitignore` before committing — GitHub blocks files over 100MB and warns above 50MB, and our dataset files are big enough to hit that limit.



<br>

## Main Tasks

<br>

### TASK 1: Build a model for Sentiment Analysis

<details>
  <summary>Click here for more details</summary>
  
  <br />

   - **Goal**: Classify customer reviews into **positive**, **negative**, or **neutral** categories to help the company improve its products and services.
   - **Task**: Develop, train, and evaluate a supervised multi-class classification model to classify the **textual content** of customer reviews as positive, negative, or neutral.

   <br>

   **Mapping Star Ratings to Sentiment Classes:**

   Since the dataset contains **star ratings (1 to 5)**, you should map them to three sentiment classes as follows:  

   | **Star Rating** | **Sentiment Class** |
   |---------------|------------------|
   |  1 - 2     | **Negative**  |
   |  3         | **Neutral**  |
   |  4 - 5     | **Positive**  |

   This is a simple approach, but you are encouraged to experiment with different mappings! 

   <br />

   **Some options:**

   You can tackle this with either traditional NLP methods or pretrained transformer models:
   - Traditional NLP: Use traditional NLP techniques for feature extraction (e.g., BoW, TF-IDF...) + a classifier (e.g., logistic regression, SVM, naive Bayes) trained from scratch.
   - Pretrained Transformers: Use a pretrained transformer-based model to leverage powerful language representations, typically via fine-tuning rather than training from scratch.

   <br />

   **Suggested Pretrained Models:**

   If you decide to use a pretrained model, here are some options:

   - **`distilbert-base-uncased`** – Lightweight and fast, ideal for limited resources.  
   - **`bert-base-uncased`** – A strong general-purpose model for sentiment analysis.  
   - **`roberta-base`** – More robust to nuanced sentiment variations.  
   - **`nlptown/bert-base-multilingual-uncased-sentiment`** – Handles multiple languages, useful for diverse datasets.  
   - **`cardiffnlp/twitter-roberta-base-sentiment`** – Optimized for short texts like social media reviews.  

   Explore models on [Hugging Face](https://huggingface.co/models) and experiment with fine-tuning to improve accuracy.

   <br />

   **Model Evaluation:**

   Evaluate the model's performance on a separate test dataset using various evaluation metrics:
   - Accuracy: Percentage of correctly classified instances.
   - Precision: Proportion of true positive predictions among all positive predictions.
   - Recall: Proportion of true positive predictions among all actual positive instances.
   - F1-score: Harmonic mean of precision and recall.

   Calculate the confusion matrix to analyze model's performance across different classes.

   <br />

   **Results:**

   Summarize the performance of your model on the held-out test dataset using both quantitative metrics and visual analysis.

   - Report the overall accuracy: Show the percentage of correctly classified test samples (X%).
   - Analyze classification performance: Present precision, recall, and F1-score for each sentiment class to provide insights into the model’s performance:
      - Class 1: Precision = X%, Recall = X%, F1-score = X%
      - Class 2: Precision = X%, Recall = X%, F1-score = X%
      - Class 3: Precision = X%, Recall = X%, F1-score = X%
   - Generate and interpret the confusion matrix: Include both a table and a visual representation to highlight correct predictions, misclassifications, and class-specific performance.

</details>


<br><br>

### TASK 2: Build a model for Product Category Clustering


<details>
  <summary>Click here for more details</summary>
  
  <br />

   - **Goal**: Simplify the dataset by clustering product categories into **4-6 meta-categories**.
   - **Task**: Develop and apply an unsupervised clustering model to group product reviews into 4–6 meaningful meta-categories based on similarities in their textual content and product characteristics.
   - **Notes**: 
      - Analyze the dataset in depth to determine the most appropriate categories.
      - After applying clustering, you can analyze the characteristics of each cluster (e.g., keywords, products, and reviews) and assign meaningful names to the identified groups to improve interpretability. For example:
         - Ebook readers
         - Batteries
         - Accessories (keyboards, laptop stands, etc.)
         - Non-electronics (Nespresso pods, pet carriers, etc.)

</details>



<br><br>

### TASK 3: Generate a summary for each product category using Generative AI

<details>
  <summary>Click here for more details</summary>
  
  <br />

   - **Goal**: Generate a summary with the reviews for each category.
   - **Task**: Create a model that generates a short article (like a blog post) for each of the product categories you created in the previous step. 

   <br />

   **Example Format**:

   For the summary of each category, you can include:

   - **Top 3 products** and key differences between them.
   - **Top complaints** for each of those products.
   - **Worst product** in the category and why it should be avoided.

   This is just an example. You can get more ideas from other consumer Reviews websites, Amazon, The Verge, The Wirecutter, etc.

   <br />

   **Some options**:

   - You can use **Pretrained Generative Models** like **T5**, or **BART** for generating coherent and well-structured summaries. These models excel at tasks like summarization and text generation, and can be fine-tuned to produce high-quality outputs based on the extracted insights from reviews.
   - You can also explore other **Transformer-based models** available on platforms like **Hugging Face**. Fine-tuning any of these pre-trained models on your specific dataset could further improve the relevance and quality of the generated summaries.
   - Another option is to use a proprietary LLM API (e.g., the OpenAI API) to generate the summaries, which can produce high-quality results. We'll explore this approach later in the course. For now, we encourage you to first experiment with a pretrained model that you can run and adapt yourself.

   <br />

   **Recommendations**:

   - If you use a pretrained model, start with the smallest versions of popular models (llama, mistral, ...). Choose a small model that you can fine tune and run fast inference on. Anywhere between 1B-8B parameters should be fine, do not go larger.
   - Work on the prompt for the summarizer by experimenting with multiple prompt variants and evaluating their performance. If prompt engineering alone does not achieve the desired quality, consider fine-tuning the model for this specific task to improve accuracy and consistency.

</details>



<br><br>

### TASK 4: Deploy the Sentiment Analysis Model


**Live demo:** [Open the public Streamlit sentiment classifier](https://project-brief-nlp-automated-customers-reviews-4z3nd6dqfyj7ba8x.streamlit.app/).
<details>
  <summary>Click here for more details</summary>
  
  <br />

   Now it's time to make your model usable by others. Deploy the sentiment classifier from Task 1 as a simple web app that anyone can try.

   - Goal: Ship a working, publicly accessible demo where users can paste a review and get a sentiment prediction.
   - Tasks:
      - Build a simple interface where a user can enter the text of a review and receive a predicted sentiment (positive, negative, or neutral), ideally with confidence scores for each class.
      - Deploy it so that it's accessible through a public URL.

   <br />

   **Recommended option: Gradio + Hugging Face Spaces**

   [Gradio](https://www.gradio.app/) lets you build an ML demo interface in a few lines of Python, and [Hugging Face Spaces](https://huggingface.co/spaces) hosts it for free. Together, they're one of the simplest ways to deploy a model.

   Notes:
   - You'll need to do some research on your own for this task (documentation and tutorials are plentiful).
   - If you prefer, you can use other tools (e.g., Streamlit, FastAPI + a hosting service), as long as the app is publicly accessible. We'll explore other deployment options later in the course.

</details>



<br><br>


## Bonus Tasks (Optional)

Your priority should be the main tasks: focus on building reliable models, trying and comparing different techniques, and getting the best possible metrics. If you have additional time, here are some extra challenges.


<br>


### Bonus 1: Visualize Your Results

<details>
  <summary>Click here for more details</summary>
  
  <br />

  - **Goal**: Turn your models' outputs into visuals that make the insights easy to explore and share.
   - For example:
      - Generate charts exploring how sentiment varies by product, review length, or time.
      - Generate a chart of the most frequent complaint themes per product or category.
      - Create an interactive dashboard (e.g. Gradio or Streamlit) showing sentiment distribution, top products, and common complaints per category.
      - ...

</details>


<br>

### Bonus 2: Translate Reviews

<details>
  <summary>Click here for more details</summary>
  
  <br />

   - **Goal**: Use a generative AI model to translate customer reviews into another language, making the review data accessible to a wider audience.

   - **Task**: Build a system that takes customer reviews written in English and generates a translation in **one target language** of your choice (e.g., Spanish, French, German, or Italian).

   - You can use:
      - A pretrained translation model from Hugging Face.
      - A generative AI model through an API.
      - Another NLP translation solution of your choice.

   - Considerations:
      - Preserve the original meaning, including positive and negative sentiment.
      - Handle informal language, abbreviations, and product-specific terminology.
      - Consider how you would handle reviews that are already written in another language.

   - **Evaluation**:
      - Manually compare a sample of translations with the original reviews.
      - Optionally, use an automatic metric such as **BLEU** or **ROUGE** if you have suitable reference translations.
      - Analyze examples where the translation works well or changes the meaning of the original review.

</details>


<br>

### Bonus 3: Your Own Bonus

<details>
  <summary>Click here for more details</summary>
  
  <br />

  You can also take this project further by adding any meaningful feature that extends the project beyond the main tasks.

</details>



<br><br>

> ⚠️ Remember: Bonuses are optional. Make sure you have completed all the mandatory tasks before spending time on additional features.



<br><br>


## Deliverables

A GitHub repository containing:

- **Source code and/or Jupyter notebooks** for all completed tasks and bonuses, including your analysis, models, experiments, and results.

- A **README.md** file with:
   - A brief description of the project.
   - Results and key findings.
   - A link to the deployed application (i.e., the URL where users can try your deployed model or app).



<br><br>


<!--

## Additional Resources


- [Machine Learning Project Structure](https://gist.github.com/luisjunco/1fa25a256ea7c5cfde2938ad6039d9fd) — A document with recommendations for organizing files and folders in a machine learning project.
   - Note: This document is designed for a project with a single model. Feel free to adapt it to your own preferences and the specific requirements of this project. For example, you could create multiple subdirectories such as `notebooks/sentiment-analysis`, `notebooks/clustering`, etc.

-->