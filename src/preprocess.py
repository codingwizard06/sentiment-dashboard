"""
preprocess.py
Text cleaning utilities used before running sentiment inference and
before keyword/topic extraction.
"""
import re
import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
MULTISPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-zA-Z0-9\s.,!?']")


def clean_text(text: str) -> str:
    """Light cleaning that keeps the text natural enough for transformer
    models (which expect fairly natural language), while stripping noise
    like URLs, @mentions and excess whitespace."""
    if not isinstance(text, str):
        return ""

    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = HASHTAG_RE.sub(r"\1", text)  # keep the word, drop the '#'
    text = NON_ALNUM_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def load_reviews(csv_path: str) -> pd.DataFrame:
    """Load a reviews CSV and apply basic cleaning + type coercion.
    Expected columns: date, product, rating, review_text
    (extra columns are preserved).
    """
    df = pd.read_csv(csv_path)

    required_cols = {"date", "product", "review_text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "review_text"])

    df["clean_text"] = df["review_text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    return df


def truncate_for_model(text: str, max_chars: int = 512) -> str:
    """Transformer models have a token limit; this is a cheap, fast guard
    so extremely long reviews don't error out at inference time."""
    return text[:max_chars]
