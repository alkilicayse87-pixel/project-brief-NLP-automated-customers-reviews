from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from scipy.special import softmax
import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'outputs' / 'task1_recommended_model.joblib'

artifact = joblib.load(MODEL_PATH)
classifier = artifact['classifier']
vectorizer = artifact['vectorizer']
labels = artifact['labels']


def predict_sentiment(review: str):
    if not review or not review.strip():
        return 'Please enter a review to classify.', {label: 0.0 for label in labels}, ''

    text = [review.strip()]
    features = vectorizer.transform(text)
    try:
        scores = classifier.decision_function(features)
        probabilities = softmax(scores, axis=1)[0]
    except AttributeError:
        probabilities = np.ones(len(labels)) / len(labels)

    label = labels[int(np.argmax(probabilities))]
    confidence = {label_name: float(prob) for label_name, prob in zip(labels, probabilities)}
    breakdown = '\n'.join(f'- {name}: {prob:.1%}' for name, prob in confidence.items())
    return label.capitalize(), confidence, breakdown


with gr.Blocks(title='Product Review Sentiment Demo') as demo:
    gr.Markdown('# Amazon Review Sentiment Classifier')
    gr.Markdown('Paste a customer review to predict whether it is positive, neutral, or negative.')

    with gr.Row():
        review_input = gr.Textbox(label='Customer review', lines=8, placeholder='Type or paste a review here...')

    predict_btn = gr.Button('Predict sentiment')
    output_label = gr.Textbox(label='Predicted sentiment')
    confidence = gr.JSON(label='Confidence by class')
    breakdown = gr.Textbox(label='Class probabilities')

    predict_btn.click(fn=predict_sentiment, inputs=[review_input], outputs=[output_label, confidence, breakdown])

if __name__ == '__main__':
    demo.launch(server_name='127.0.0.1', server_port=7860, share=False)
