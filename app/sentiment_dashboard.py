import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
model_path = os.path.join(ROOT_DIR, "models", "sentiment_model.pkl")
vectorizer_path = os.path.join(ROOT_DIR, "models", "vectorizer.pkl")
data_path = os.path.join(ROOT_DIR, "data", "Amazon Product Review.txt")

model = joblib.load(model_path)
tfidf_vectorizer = joblib.load(vectorizer_path)

st.title("🌟 Product Review Sentiment Analysis 🌟")

st.header("✍️ Input Your Review")
user_review = st.text_area("Enter your product review:", "")

if st.button("Predict Sentiment"):
    if user_review:
        review_vector = tfidf_vectorizer.transform([user_review])
        prediction = model.predict(review_vector)
        sentiment_map = {0: "Negative (1-2 stars)", 1: "Neutral (3 stars)", 2: "Positive (4-5 stars)"}
        predicted_sentiment = sentiment_map[prediction[0]]
        st.success(f"The predicted sentiment is: **{predicted_sentiment}**")
    else:
        st.warning("Please enter a review before predicting.")

if os.path.exists(data_path):
    raw_data = pd.read_csv(data_path)
    sentiment_mapping = {0: "Negative", 1: "Positive"}
    if "sentiment" in raw_data.columns:
        raw_data["Sentiment"] = raw_data["sentiment"].map(sentiment_mapping)
        raw_data = raw_data.dropna(subset=["Sentiment"])
    else:
        raw_data["Sentiment"] = np.nan
else:
    raw_data = pd.DataFrame(columns=["Sentiment", "review_body"])

st.header("📊 Sentiment Distribution")
if not raw_data.empty and "Sentiment" in raw_data.columns and raw_data["Sentiment"].notna().any():
    sentiment_count = raw_data["Sentiment"].value_counts()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=sentiment_count.index, y=sentiment_count.values, palette="viridis")
    plt.title("Distribution of Sentiments", fontsize=18, fontweight="bold")
    plt.xlabel("Sentiment", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    st.pyplot(plt)
else:
    st.info("Sentiment distribution is not available because the dataset is missing or does not contain sentiment labels.")

st.header("☁️ Word Cloud of Reviews")
if not raw_data.empty and "review_body" in raw_data.columns:
    text_data = " ".join(raw_data["review_body"].astype(str).tolist())
    if text_data.strip():
        wordcloud = WordCloud(width=800, height=400, background_color="black", colormap="plasma").generate(text_data)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    else:
        st.info("Not enough review text available to generate a word cloud.")
else:
    st.info("Review text is not available for generating a word cloud.")
