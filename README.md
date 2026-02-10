# Product Review Sentiment Analysis

Ever wondered what customers *really* think? This project reads product reviews and figures out whether they're positive, negative, or neutral—so you can spot patterns, fix what's broken, and double down on what works.

![Dashboard screenshot](Screenshot%20(130).png)

---

## What You Can Do

- **Analyze any review** — Paste text and get an instant sentiment result (positive, neutral, or negative)
- **Try sample reviews** — One-click examples to see how it works
- **Explore insights** — Word clouds and sentiment charts from your dataset
- **Keep it simple** — Clean dashboard, no clutter

---

## Quick Start

```bash
# Clone and go
git clone https://github.com/yourusername/Product-Review-Sentiment-Analysis.git
cd Product-Review-Sentiment-Analysis

# Set up a virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app/sentiment_dashboard.py
```

Open the URL in your browser (usually `http://localhost:8501`) and you're good to go.

---

## Tech Stack

- **Python** — Core logic
- **Streamlit** — Web dashboard
- **Scikit-learn** — Sentiment model
- **NLTK** — Text processing
- **Pandas, NumPy** — Data handling
- **Matplotlib, Seaborn, WordCloud** — Visualizations

---

## Project Structure

```
├── app/
│   └── sentiment_dashboard.py   # The Streamlit app
├── data/
│   └── Amazon Product Review.txt # Sample dataset
├── models/
│   ├── sentiment_model.pkl      # Trained model
│   └── vectorizer.pkl           # TF-IDF vectorizer
└── notebooks/                    # Analysis notebooks
```

---

## Contributing

Ideas and pull requests are welcome. If you’d like to contribute:

1. Fork the repo
2. Create a branch (`git checkout -b feature/your-idea`)
3. Commit your changes (`git commit -m 'Add your idea'`)
4. Push and open a pull request

---

## License

This project is licensed under the MIT License.

---

## Contact

Questions or feedback? Reach out via [the author's portfolio](https://nafisalawalidris.github.io/13/).

---

If this project helps you, consider giving it a star ⭐ — it goes a long way.
