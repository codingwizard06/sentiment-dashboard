"""
insights.py
Derives the analytical layer on top of raw sentiment predictions:
- time trends
- per-product breakdowns
- top keywords driving negative sentiment (TF-IDF based)
- simple spike-alert detection
"""
from typing import List, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# A small stopword set kept local so the project has zero extra downloads
# (avoids requiring nltk's separate corpus download step).
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in", "on", "for", "with", "at", "by",
    "this", "that", "it", "its", "i", "my", "me", "we", "our", "you",
    "your", "they", "them", "their", "he", "she", "his", "her", "as",
    "so", "if", "not", "no", "very", "just", "did", "do", "does", "have",
    "has", "had", "will", "would", "can", "could", "should", "than",
    "then", "there", "here", "also", "too", "up", "down", "out", "into",
}


def sentiment_over_time(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Returns counts of each sentiment bucketed by time period (default weekly)."""
    grouped = (
        df.set_index("date")
          .groupby([pd.Grouper(freq=freq), "sentiment"])
          .size()
          .reset_index(name="count")
    )
    return grouped


def sentiment_by_product(df: pd.DataFrame) -> pd.DataFrame:
    """Returns sentiment counts per product, plus a % positive column for ranking."""
    counts = df.groupby(["product", "sentiment"]).size().unstack(fill_value=0)
    for col in ["positive", "neutral", "negative"]:
        if col not in counts.columns:
            counts[col] = 0
    counts["total"] = counts[["positive", "neutral", "negative"]].sum(axis=1)
    counts["pct_positive"] = (counts["positive"] / counts["total"] * 100).round(1)
    counts["pct_negative"] = (counts["negative"] / counts["total"] * 100).round(1)
    return counts.reset_index().sort_values("pct_negative", ascending=False)


def top_keywords(df: pd.DataFrame, sentiment: str = "negative", top_n: int = 15) -> List[Tuple[str, float]]:
    """Uses TF-IDF over the subset of reviews with the given sentiment label
    to surface the terms that most distinguish those reviews — a fast,
    dependency-light stand-in for full topic modeling."""
    subset = df[df["sentiment"] == sentiment]["clean_text"]
    if len(subset) < 3:
        return []

    vectorizer = TfidfVectorizer(
        max_features=200,
        stop_words=list(STOPWORDS),
        ngram_range=(1, 2),
        min_df=2,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(subset)
    except ValueError:
        return []

    scores = tfidf_matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def detect_spikes(df: pd.DataFrame, freq: str = "W", threshold_pct: float = 40.0) -> pd.DataFrame:
    """Flags time periods where negative-sentiment share exceeds threshold_pct.
    Useful as a simple 'alert' signal for a brand-monitoring use case."""
    grouped = df.set_index("date").groupby(pd.Grouper(freq=freq))
    rows = []
    for period, g in grouped:
        if len(g) == 0:
            continue
        neg_pct = (g["sentiment"] == "negative").mean() * 100
        rows.append({"period": period, "total_reviews": len(g), "pct_negative": round(neg_pct, 1)})

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["alert"] = result["pct_negative"] >= threshold_pct
    return result
