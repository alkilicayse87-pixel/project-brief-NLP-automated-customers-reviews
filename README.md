![logo_ironhack_blue](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Project | Business Case: Automated Customer Reviews

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
- You'll be working with realistic, real-world datasets.
- Spend some time exploring and understanding the dataset. You may need to fix data quality issues, discard irrelevant features, handle missing values, and make other preprocessing decisions before training your model.
- Add your `datasets` folder (or at least the CSV files in it) to `.gitignore` before committing — GitHub blocks files over 100MB and warns above 50MB, and our dataset files are big enough to hit that limit.



<br>

## Main Tasks

<br>

### 1. Build a model for Sentiment Analysis

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


**Model Building:**

For classifying customer reviews into **positive, negative, or neutral**, use **pretrained transformer-based models** to leverage powerful language representations without training from scratch.  

**Suggested Pretrained Models:**

- **`distilbert-base-uncased`** – Lightweight and fast, ideal for limited resources.  
- **`bert-base-uncased`** – A strong general-purpose model for sentiment analysis.  
- **`roberta-base`** – More robust to nuanced sentiment variations.  
- **`nlptown/bert-base-multilingual-uncased-sentiment`** – Handles multiple languages, useful for diverse datasets.  
- **`cardiffnlp/twitter-roberta-base-sentiment`** – Optimized for short texts like social media reviews.  

Explore models on [Hugging Face](https://huggingface.co/models) and experiment with fine-tuning to improve accuracy.

**Model Evaluation:**

Evaluate the model's performance on a separate test dataset using various evaluation metrics:
- Accuracy: Percentage of correctly classified instances.
- Precision: Proportion of true positive predictions among all positive predictions.
- Recall: Proportion of true positive predictions among all actual positive instances.
- F1-score: Harmonic mean of precision and recall.

Calculate the confusion matrix to analyze model's performance across different classes.


**Results:**

Summarize the performance of your model on the held-out test dataset using both quantitative metrics and visual analysis.

- Report the overall accuracy: Show the percentage of correctly classified test samples (X%).
- Analyze classification performance: Present precision, recall, and F1-score for each sentiment class to provide insights into the model’s performance:
   - Class 1: Precision = X%, Recall = X%, F1-score = X%
   - Class 2: Precision = X%, Recall = X%, F1-score = X%
   - Class 3: Precision = X%, Recall = X%, F1-score = X%
- Generate and interpret the confusion matrix: Include both a table and a visual representation to highlight correct predictions, misclassifications, and class-specific performance.

<br><br>

### 2. Build a model for Product Category Clustering

- **Goal**: Simplify the dataset by clustering product categories into **4-6 meta-categories**.
- **Task**: Develop and apply an unsupervised clustering model to group product reviews into 4–6 meaningful meta-categories based on similarities in their textual content and product characteristics.
- **Notes**: 
   - Analyze the dataset in depth to determine the most appropriate categories.
   - After applying clustering, you can analyze the characteristics of each cluster (e.g., keywords, products, and reviews) and assign meaningful names to the identified groups to improve interpretability. For example:
      - Ebook readers
      - Batteries
      - Accessories (keyboards, laptop stands, etc.)
      - Non-electronics (Nespresso pods, pet carriers, etc.)


<br><br>

### 3. Generate a summary for each product category using Generative AI

- **Goal**: Generate a summary with the reviews for each category.
- **Task**: Create a model that generates a short article (like a blog post) for each of the product categories you created in the previous step. 


**Example Format**:

For the summary of each category, you can include:

- **Top 3 products** and key differences between them.
- **Top complaints** for each of those products.
- **Worst product** in the category and why it should be avoided.

This is just an example. You can get more ideas from other consumer Reviews websites, Amazon, The Verge, The Wirecutter, etc.

**Some options**:

- You can use **Pretrained Generative Models** like **T5**, or **BART** for generating coherent and well-structured summaries. These models excel at tasks like summarization and text generation, and can be fine-tuned to produce high-quality outputs based on the extracted insights from reviews.
- You can also explore other **Transformer-based models** available on platforms like **Hugging Face**. Fine-tuning any of these pre-trained models on your specific dataset could further improve the relevance and quality of the generated summaries.
- Another option is to use a proprietary LLM API (e.g., the OpenAI API) to generate the summaries, as it can produce high-quality results. However, we encourage you to first explore using a pretrained model that you can run and adapt yourself.


**Recommendations**:

- If you use a pretrained model, start with the smallest versions of popular models (llama, mistral, ...). Choose a small model that you can fine tune and run fast inference on. Anywhere between 1B-8B parameters should be fine, do not go larger.
- Work on the prompt for the summarizer by experimenting with multiple prompt variants and evaluating their performance. If prompt engineering alone does not achieve the desired quality, consider fine-tuning the model for this specific task to improve accuracy and consistency.


<br><br>

### 4. Build & Deploy a final product

Now it's time to turn your models into something people can actually use. Bring together the sentiment classifier, clustering model, and summarization model you've built into a single, functional product.

- **Goal**: Ship your work as a web app or website that delivers real value to end users, turning your models into a tool stakeholders could actually use to make decisions.

- **Tasks**:
   - Plan and build a web app or website that integrates some of the functionality you've build.
   - Deploy it so it's accessible to others.


**Some Ideas**:

We provide you with some ideas below. However, you are not limited to these options. Feel free to build a web app or website that does different things to what listed below.

1. **Create a website where users can classify a review and view category summaries**. One part of the website would allow users to enter the text of a review and perform sentiment analysis (i.e., get a prediction of whether the review is positive, negative, or neutral). Another part of the website would allow users to browse the summaries you have generated for each category.
2. **Create a website for the marketing department in your company** to gain insights on how well the products are received by customers. For example, users in your webpage can choose between product categories and be shown statistics insights (distribution of ratings, best product ratings, etc), and text summarization for that specific category (which are the best product in this category, etc).
3. **Build a live review aggregator**: this could be a website like, for example, https://www.trustpilot.com/ or https://www.yelp.com/, organizing reviews strategically for buyers. You could add functionality for users to add reviews (for example, through a form, a user could write about a product, selecting which cluster category it belongs to and the rating given). Once a review is submitted, it could be displayed on the page as a ‘recently added review’. Feel free to come up with your own ideas about how you would like your live review aggregator to look like and behave.
4. **Develop a website that generates recommendations by allowing users to upload a csv file with reviews**. For example, this website could allow business owners to upload a dataset of their products and respective reviews. Your website would process these, classifying them, clustering them, and showing insights in the form of small articles listing top products, main product issues, etc., for example (e.g., a list of articles, one per product; a list of articles, one per cluster).
5. **Develop a website that allows users to search for information about a product or product category through a text box**. This could be a text box where users type in what they are looking for / would like to buy. The output could display recommendations of products in text summary format, the category of the product, and the sentiment distribution for that product.


<br>

## Suggested Workflow

1. **Data Collection**: Gather and preprocess the dataset(s).
2. **Model Development**:
   - Create and evaluate the review classification model.
   - Create and test the clustering model.
   - Create and test the summarization model.
3. **Deployment**: Deploy the models using your chosen framework.
4. **Documentation**: Prepare the README, report, and presentation.
5. **Final Delivery**: Submit all deliverables, including the deployed app and final output.

