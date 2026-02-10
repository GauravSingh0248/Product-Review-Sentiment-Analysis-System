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

# Load model once
@st.cache_resource
def load_model():
    return joblib.load(model_path), joblib.load(vectorizer_path)

model, tfidf_vectorizer = load_model()

# Sample reviews for quick try
SAMPLE_REVIEWS = {
    "😊 Positive": "Great product! Exactly what I needed. Fast delivery and excellent quality. Will buy again!",
    "😐 Neutral": "It works as expected. Nothing special, but it does the job fine.",
    "😞 Negative": "Terrible experience. Broke after one day. Waste of money. Do not recommend.",
}

# Custom styling
st.set_page_config(page_title="Sentiment Analysis", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .main { padding: 1rem 2rem; }
    .stMetric { background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%); 
                padding: 1rem; border-radius: 12px; border: 1px solid #667eea44; }
    .sentiment-box { padding: 1rem 1.5rem; border-radius: 12px; font-size: 1.1rem; 
                     font-weight: 600; text-align: center; margin: 1rem 0; }
    .positive { background: linear-gradient(135deg, #10b98122, #34d39933); color: #047857; }
    .neutral { background: linear-gradient(135deg, #f59e0b22, #fbbf2433); color: #b45309; }
    .negative { background: linear-gradient(135deg, #ef444422, #f8717133); color: #b91c1c; }
    div[data-testid="stHorizontalBlock"] > div { gap: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Product Review Sentiment")
st.caption("Analyze how customers feel about products from their reviews")

# Sidebar navigation
page = st.sidebar.radio(
    "Navigate",
    ["✨ Try it", "📈 Insights"],
    label_visibility="collapsed"
)

if page == "✨ Try it":
    st.header("Analyze a Review")
    
    # Quick samples - click to fill
    st.write("**Try a sample:**")
    cols = st.columns(3)
    for i, (label, text) in enumerate(SAMPLE_REVIEWS.items()):
        with cols[i]:
            if st.button(label, key=f"sample_{i}", use_container_width=True):
                st.session_state.user_review = text
                st.rerun()
    
    user_review = st.text_area(
        "Or paste your own review:",
        key="user_review",
        height=100,
        placeholder="e.g., This product exceeded my expectations..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        predict_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    if predict_btn:
        if user_review.strip():
            review_vector = tfidf_vectorizer.transform([user_review])
            prediction = model.predict(review_vector)[0]
            sentiment_map = {
                0: ("Negative (1–2 stars)", "negative", "😞"),
                1: ("Neutral (3 stars)", "neutral", "😐"),
                2: ("Positive (4–5 stars)", "positive", "😊"),
            }
            label, css_class, emoji = sentiment_map[prediction]
            st.session_state.last_result = (label, css_class, emoji)
        else:
            st.warning("Please enter a review first.")
    
    if "last_result" in st.session_state:
        label, css_class, emoji = st.session_state.last_result
        st.markdown(f'<div class="sentiment-box {css_class}">{emoji} {label}</div>', unsafe_allow_html=True)

else:
    # Insights page
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
    
    has_data = not raw_data.empty and "Sentiment" in raw_data.columns and raw_data["Sentiment"].notna().any()
    
    if has_data:
        sentiment_count = raw_data["Sentiment"].value_counts()
        
        # Metrics row
        st.subheader("Overview")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Reviews", f"{len(raw_data):,}")
        with m2:
            pos = sentiment_count.get("Positive", 0)
            st.metric("Positive", f"{pos:,}")
        with m3:
            neg = sentiment_count.get("Negative", 0)
            st.metric("Negative", f"{neg:,}")
        
        # Charts in expanders
        with st.expander("📊 Sentiment Distribution", expanded=True):
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ["#ef4444", "#10b981"]
            sns.barplot(x=sentiment_count.index, y=sentiment_count.values, palette=colors, ax=ax)
            ax.set_title("Reviews by Sentiment")
            ax.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with st.expander("☁️ Word Cloud", expanded=False):
            if "review_body" in raw_data.columns:
                text_data = " ".join(raw_data["review_body"].astype(str).str.replace("<br />", " ", regex=False).tolist())
                if text_data.strip():
                    wordcloud = WordCloud(width=800, height=400, background_color="white", 
                                         colormap="viridis").generate(text_data)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("Not enough text for a word cloud.")
    else:
        st.info("Load a dataset with sentiment labels to see insights here.")
