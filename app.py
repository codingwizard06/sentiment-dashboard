"""
app.py
Sentiment / Review Analysis Dashboard
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px

from src.preprocess import load_reviews
from src.sentiment_model import SentimentModel
from src.insights import sentiment_over_time, sentiment_by_product, top_keywords, detect_spikes
from src.utils import kpi_summary, filter_dataframe

st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
)

DATA_PATH = "data/reviews.csv"


# ----------------------------------------------------------------------
# Cached loaders — model + inference results only recompute when the
# underlying file or model choice changes, which keeps the UI responsive.
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading sentiment model (first run only)...")
def get_model(model_name: str):
    return SentimentModel(model_name=model_name)


@st.cache_data(show_spinner="Running sentiment inference on reviews...")
def get_scored_data(csv_path: str, model_name: str) -> pd.DataFrame:
    df = load_reviews(csv_path)
    model = get_model(model_name)
    scored = model.predict_dataframe(df)
    return scored


def main():
    st.title("📊 Sentiment & Review Analysis Dashboard")
    st.caption(
        "Upload customer reviews, classify sentiment with a Hugging Face "
        "transformer model, and explore trends, product breakdowns and "
        "what's driving negative feedback."
    )

    # ------------------------------------------------------------------
    # Sidebar: data source + model choice
    # ------------------------------------------------------------------
    st.sidebar.header("⚙️ Settings")

    uploaded_file = st.sidebar.file_uploader(
        "Upload a reviews CSV (columns: date, product, review_text)",
        type=["csv"],
    )

    model_choice = st.sidebar.selectbox(
        "Sentiment model",
        options=[
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "distilbert-base-uncased-finetuned-sst-2-english",
        ],
        help="The Cardiff NLP model returns positive/neutral/negative and "
             "handles informal review text well. DistilBERT is faster but "
             "binary only (positive/negative).",
    )

    data_path = DATA_PATH
    if uploaded_file is not None:
        with open("data/_uploaded.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        data_path = "data/_uploaded.csv"

    with st.spinner("Loading and scoring reviews..."):
        try:
            df = get_scored_data(data_path, model_choice)
        except Exception as e:
            st.error(f"Failed to load/score data: {e}")
            st.stop()

    # ------------------------------------------------------------------
    # Sidebar: filters
    # ------------------------------------------------------------------
    st.sidebar.header("🔍 Filters")

    all_products = sorted(df["product"].unique().tolist())
    selected_products = st.sidebar.multiselect("Product", all_products, default=all_products)

    min_date, max_date = df["date"].min(), df["date"].max()
    date_range = st.sidebar.date_input(
        "Date range", value=(min_date.date(), max_date.date()),
        min_value=min_date.date(), max_value=max_date.date(),
    )

    sentiment_options = sorted(df["sentiment"].unique().tolist())
    selected_sentiments = st.sidebar.multiselect(
        "Sentiment", sentiment_options, default=sentiment_options
    )

    filtered = filter_dataframe(df, selected_products, date_range, selected_sentiments)

    if filtered.empty:
        st.warning("No reviews match the current filters.")
        st.stop()

    # ------------------------------------------------------------------
    # KPI cards
    # ------------------------------------------------------------------
    kpis = kpi_summary(filtered)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", kpis["total"])
    col2.metric("% Positive", f"{kpis['pct_positive']}%")
    col3.metric("% Negative", f"{kpis['pct_negative']}%")
    col4.metric("% Neutral", f"{kpis['pct_neutral']}%")

    st.divider()

    # ------------------------------------------------------------------
    # Trend + breakdown charts
    # ------------------------------------------------------------------
    left, right = st.columns(2)

    with left:
        st.subheader("Sentiment Trend Over Time")
        freq = st.radio("Granularity", ["D", "W", "M"], index=1, horizontal=True,
                         format_func=lambda x: {"D": "Daily", "W": "Weekly", "M": "Monthly"}[x])
        trend_df = sentiment_over_time(filtered, freq=freq)
        fig_trend = px.line(
            trend_df, x="date", y="count", color="sentiment",
            color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"},
            markers=True,
        )
        fig_trend.update_layout(height=380, legend_title_text="")
        st.plotly_chart(fig_trend, use_container_width=True)

    with right:
        st.subheader("Sentiment by Product")
        product_df = sentiment_by_product(filtered)
        fig_product = px.bar(
            product_df, x="product", y=["positive", "neutral", "negative"],
            color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"},
            barmode="stack",
        )
        fig_product.update_layout(height=380, legend_title_text="", xaxis_title="")
        st.plotly_chart(fig_product, use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------
    # Keyword insights + alerts
    # ------------------------------------------------------------------
    left2, right2 = st.columns(2)

    with left2:
        st.subheader("🔑 Top Keywords in Negative Reviews")
        keywords = top_keywords(filtered, sentiment="negative", top_n=15)
        if keywords:
            kw_df = pd.DataFrame(keywords, columns=["keyword", "score"])
            fig_kw = px.bar(kw_df.sort_values("score"), x="score", y="keyword", orientation="h")
            fig_kw.update_layout(height=420, yaxis_title="", xaxis_title="TF-IDF score")
            st.plotly_chart(fig_kw, use_container_width=True)
        else:
            st.info("Not enough negative reviews in the current filter to extract keywords.")

    with right2:
        st.subheader("🚨 Spike Alerts (negative sentiment)")
        threshold = st.slider("Alert threshold (% negative)", 10, 90, 40, step=5)
        spikes = detect_spikes(filtered, freq="W", threshold_pct=threshold)
        if not spikes.empty:
            fig_alert = px.bar(
                spikes, x="period", y="pct_negative", color="alert",
                color_discrete_map={True: "#e74c3c", False: "#95a5a6"},
            )
            fig_alert.add_hline(y=threshold, line_dash="dash", line_color="black")
            fig_alert.update_layout(height=420, showlegend=False, xaxis_title="", yaxis_title="% negative")
            st.plotly_chart(fig_alert, use_container_width=True)

            flagged = spikes[spikes["alert"]]
            if not flagged.empty:
                st.error(f"⚠️ {len(flagged)} week(s) exceeded the {threshold}% negative-sentiment threshold.")
        else:
            st.info("Not enough data to compute weekly alerts for this filter.")

    st.divider()

    # ------------------------------------------------------------------
    # Drill-down table
    # ------------------------------------------------------------------
    st.subheader("📋 Review Detail (drill-down)")
    show_only_negative = st.checkbox("Show only negative reviews", value=False)
    table_df = filtered[filtered["sentiment"] == "negative"] if show_only_negative else filtered

    st.dataframe(
        table_df[["date", "product", "review_text", "sentiment", "sentiment_score"]]
        .sort_values("date", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
        height=350,
    )

    csv_export = table_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered data as CSV", csv_export, "filtered_reviews.csv", "text/csv")


if __name__ == "__main__":
    main()
