"""
utils.py
Small shared helpers used by the Streamlit app.
"""
import pandas as pd


def kpi_summary(df: pd.DataFrame) -> dict:
    """Returns high-level KPI numbers shown at the top of the dashboard."""
    total = len(df)
    if total == 0:
        return {"total": 0, "pct_positive": 0, "pct_negative": 0, "pct_neutral": 0}

    counts = df["sentiment"].value_counts(normalize=True) * 100
    return {
        "total": total,
        "pct_positive": round(counts.get("positive", 0), 1),
        "pct_negative": round(counts.get("negative", 0), 1),
        "pct_neutral": round(counts.get("neutral", 0), 1),
    }


def filter_dataframe(df: pd.DataFrame, products: list, date_range, sentiments: list) -> pd.DataFrame:
    """Applies the sidebar filters selected in the Streamlit UI."""
    filtered = df.copy()

    if products:
        filtered = filtered[filtered["product"].isin(products)]

    if date_range and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered = filtered[(filtered["date"] >= start) & (filtered["date"] <= end)]

    if sentiments:
        filtered = filtered[filtered["sentiment"].isin(sentiments)]

    return filtered
