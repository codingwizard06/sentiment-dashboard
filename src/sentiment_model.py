"""
sentiment_model.py
Wraps a Hugging Face transformers pipeline for sentiment classification
and runs it efficiently over a pandas DataFrame in batches.
"""
from typing import List
import pandas as pd
from transformers import pipeline

from .preprocess import truncate_for_model

# Good default: trained specifically on social-media / informal review-style
# text and returns positive / neutral / negative (3-class), which maps
# better onto real customer reviews than a plain binary model.
DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Lighter/faster fallback if the above is too large for your machine.
# Binary only: POSITIVE / NEGATIVE.
FALLBACK_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

_LABEL_MAP = {
    # cardiffnlp labels
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
    # distilbert sst-2 labels
    "POSITIVE": "positive",
    "NEGATIVE": "negative",
}


class SentimentModel:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: int = -1):
        """
        model_name: any HF text-classification model, defaults to a model
                    tuned for informal/social text.
        device: -1 for CPU, 0 (or GPU index) for CUDA.
        """
        self.model_name = model_name
        try:
            self.pipe = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=device,
                truncation=True,
            )
        except Exception as e:
            # Graceful fallback so the app doesn't crash if the primary
            # model can't be downloaded (e.g. restricted network).
            print(f"[sentiment_model] Could not load {model_name} ({e}); "
                  f"falling back to {FALLBACK_MODEL}")
            self.model_name = FALLBACK_MODEL
            self.pipe = pipeline(
                "sentiment-analysis",
                model=FALLBACK_MODEL,
                tokenizer=FALLBACK_MODEL,
                device=device,
                truncation=True,
            )

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[dict]:
        texts = [truncate_for_model(t) for t in texts]
        results = self.pipe(texts, batch_size=batch_size)
        normalized = []
        for r in results:
            label = _LABEL_MAP.get(r["label"], r["label"].lower())
            normalized.append({"label": label, "score": float(r["score"])})
        return normalized

    def predict_dataframe(self, df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
        """Adds `sentiment` and `sentiment_score` columns to a copy of df."""
        df = df.copy()
        texts = df[text_col].tolist()

        preds_label = []
        preds_score = []
        # Simple batching to keep memory bounded on large CSVs.
        batch_size = 32
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            preds = self.predict_batch(batch, batch_size=batch_size)
            preds_label.extend([p["label"] for p in preds])
            preds_score.extend([p["score"] for p in preds])

        df["sentiment"] = preds_label
        df["sentiment_score"] = preds_score
        return df
