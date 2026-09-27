# 📊 Sentiment & Review Analysis Dashboard

An interactive dashboard that classifies customer reviews by sentiment using a
Hugging Face transformer model, then surfaces trends, product-level
breakdowns, keyword drivers of negative sentiment, and spike alerts — the
kind of tool used in marketing, product, and brand-monitoring teams to keep
a pulse on customer feedback.

Built with **Hugging Face Transformers**, **pandas**, **scikit-learn**, and
**Streamlit + Plotly**.

---

## Features

- **Sentiment classification** on customer reviews using a pretrained
  RoBERTa model fine-tuned on social/review-style text
  (positive / neutral / negative), with a lighter DistilBERT model
  available as a fallback.
- **Trend charts** — sentiment volume over time (daily/weekly/monthly).
- **Product breakdown** — stacked sentiment bars per product, ranked by
  % negative.
- **Keyword extraction** — TF-IDF based keyword/phrase ranking over
  negative reviews, so you can see *what* people are complaining about,
  not just *that* they are.
- **Spike alerts** — flags time windows where negative sentiment crosses
  a configurable threshold.
- **Drill-down table** with CSV export of the filtered view.
- **CSV upload** — bring your own reviews (must include `date`, `product`,
  `review_text` columns) or use the bundled synthetic sample dataset.

---

## Project Structure

```
sentiment-dashboard/
├── data/
│   └── reviews.csv            # synthetic sample dataset (600 reviews)
├── src/
│   ├── __init__.py
│   ├── preprocess.py          # text cleaning + CSV loading
│   ├── sentiment_model.py     # HF pipeline wrapper + batch inference
│   ├── insights.py            # trends, product breakdown, keywords, alerts
│   └── utils.py                # KPI calc + filter helpers
├── app.py                      # Streamlit dashboard (entry point)
├── generate_sample_data.py     # regenerates the synthetic dataset
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd sentiment-dashboard

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

The first run will download the sentiment model from Hugging Face
(a few hundred MB) — this is cached locally afterwards, so subsequent
runs start instantly.

The app opens at `http://localhost:8501`.

---

## Using Your Own Data

Upload a CSV from the sidebar with at minimum these columns:

| column        | description                          |
|---------------|---------------------------------------|
| `date`        | review date (any pandas-parseable format) |
| `product`     | product/brand name                    |
| `review_text` | the review/comment text               |

An optional `rating` column is preserved but not required.

To regenerate the bundled synthetic dataset (e.g. with a different size):

```bash
python generate_sample_data.py
```

> **Note:** The bundled `data/reviews.csv` is synthetically generated
> (see `generate_sample_data.py`) so the project runs immediately without
> needing an external download. To use real data, point the CSV upload at
> an export from Amazon/Yelp/Twitter datasets (e.g. from Kaggle) formatted
> to the columns above.

---

## Design Notes / Trade-offs

- **Model choice**: `cardiffnlp/twitter-roberta-base-sentiment-latest` was
  chosen over a plain binary SST-2 model because customer reviews are
  informal and frequently neutral — a 3-class model gives more actionable
  signal than forcing every neutral review into positive/negative. The
  trade-off is a larger model and marginally slower inference; DistilBERT
  is offered in the sidebar as a faster, binary alternative when latency
  matters more than granularity.
- **Keyword extraction**: TF-IDF (via scikit-learn) was used instead of a
  full topic-modeling library (e.g. BERTopic) to keep the dependency
  footprint small and inference fast — appropriate for a dashboard that
  needs to feel responsive on every filter change. It's a reasonable
  stand-in for lightweight aspect discovery; a production system handling
  large volumes would likely upgrade to BERTopic or an LLM-based
  summarizer for higher-quality themes.
- **Caching**: `st.cache_resource` is used for the loaded model (expensive,
  load-once) and `st.cache_data` for the scored DataFrame (recomputed only
  when the input file or model changes), keeping the UI responsive when
  users adjust filters.

## Possible Extensions

- Aspect-based sentiment (e.g. "battery life: negative" vs "camera: positive")
- Live data ingestion via Twitter/X API or a Google Reviews connector
- Deploy to Streamlit Community Cloud or Hugging Face Spaces
- Swap TF-IDF keywording for BERTopic or an LLM-based theme summarizer
- Add authentication and a scheduled ingestion job for a production setup

## License

MIT — free to use and adapt.
