"""
Streamlit demo: LevelEdge regression price-ratio predictor.

Run with:
    streamlit run regression_demo.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from leveledge.constants import ALLOWED_INTERVALS, US_EASTERN
from leveledge.predictor import Predictor

st.set_page_config(page_title="LevelEdge Regression Demo", layout="wide")
st.title("LevelEdge — Regression Price Predictor")
st.caption("Predicts the future close / current close ratio using XGBRegressor.")

# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Parameters")

    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    interval = st.selectbox("Interval", ALLOWED_INTERVALS, index=ALLOWED_INTERVALS.index("1h"))
    st.divider()
    st.subheader("Target date & time (ET)")
    default_target = datetime.now(tz=US_EASTERN) + timedelta(days=7)
    target_date = st.date_input("Date", value=default_target.date())
    target_time = st.time_input("Time", value=time(16, 0))

    run = st.button("Run Prediction", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Run prediction
# ---------------------------------------------------------------------------
if run:
    target_dt = datetime.combine(target_date, target_time, tzinfo=US_EASTERN)

    with st.spinner("Fetching data and training regression model..."):
        try:
            predictor = Predictor(
                ticker_str=ticker,
                target_datetime=target_dt,
                interval=interval,
                price=1.0,  # unused by regression; satisfies constructor signature
            )
            predictor.train_regression()
            ratio = predictor.predict_regression()
            mae, rmse, r2 = predictor.get_regression_model_metrics()

        except ValueError as e:
            st.error(f"Input error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    current_price = predictor.current_price
    predicted_price = current_price * ratio
    pct_move = (ratio - 1) * 100
    candles_ahead = predictor.candles_ahead

    # ---------------------------------------------------------------------------
    # Prediction result banner
    # ---------------------------------------------------------------------------
    direction = "UP" if ratio >= 1 else "DOWN"
    color = "normal" if ratio >= 1 else "inverse"

    st.subheader("Prediction")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${current_price:.2f}")
    col2.metric("Predicted Price", f"${predicted_price:.2f}", delta=f"{pct_move:+.2f}%", delta_color=color)
    col3.metric("Predicted Ratio", f"{ratio:.4f}")
    col4.metric("Candles Ahead", str(candles_ahead))

    # ---------------------------------------------------------------------------
    # Model metrics
    # ---------------------------------------------------------------------------
    st.divider()
    st.subheader("Regression Model Metrics (walk-forward avg)")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae * 100:.4f}%",
              help="Mean Absolute Error expressed as % of current price (target is a ratio, so ×100).")
    m2.metric("RMSE", f"{rmse * 100:.4f}%",
              help="Root Mean Squared Error expressed as % of current price (target is a ratio, so ×100).")
    r2_label = f"{r2:.4f}"
    r2_help = (
        "Coefficient of determination. "
        "Negative means the model predicts worse than simply guessing the mean ratio every time — "
        "common for financial time series where future returns are near-random."
    )
    m3.metric("R²", r2_label, delta="below baseline" if r2 < 0 else "above baseline",
              delta_color="inverse" if r2 < 0 else "normal", help=r2_help)

    # ---------------------------------------------------------------------------
    # Price chart
    # ---------------------------------------------------------------------------
    st.divider()
    st.subheader("Recent Price History")

    hist = predictor.data_withna[["Datetime", "Open", "High", "Low", "Close"]].copy()
    hist["Datetime"] = pd.to_datetime(hist["Datetime"])

    # Restrict chart to today's candles only
    today = hist["Datetime"].dt.normalize().iloc[-1]
    hist_today = hist[hist["Datetime"].dt.normalize() == today]
    if hist_today.empty:
        hist_today = hist  # fallback: show all data

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # Candlesticks (today only)
    for _, row in hist_today.iterrows():
        color = "#26a69a" if row["Close"] >= row["Open"] else "#ef5350"
        ax.plot([row["Datetime"], row["Datetime"]], [row["Low"], row["High"]],
                color=color, linewidth=0.8)
        ax.bar(row["Datetime"], row["Close"] - row["Open"], bottom=row["Open"],
               color=color, width=pd.Timedelta(minutes=predictor.interval_min) * 0.6,
               align="center")

    # Fit y-axis to today's price range with a small pad
    y_min = hist_today["Low"].min()
    y_max = hist_today["High"].max()
    y_pad = (y_max - y_min) * 0.1 or 1.0
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.axhline(predicted_price, color="#9c6dff", linestyle=":", linewidth=1.5,
               label=f"Predicted ${predicted_price:.2f}")

    ax.set_xlabel("Datetime", color="white")
    ax.set_ylabel("Price ($)", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.legend(facecolor="#1e1e1e", labelcolor="white", framealpha=0.8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # ---------------------------------------------------------------------------
    # Raw data preview
    # ---------------------------------------------------------------------------
    with st.expander("Raw feature data (last 20 rows)"):
        display_cols = ["Datetime", "Open", "High", "Low", "Close", "Volume",
                        "Future_close_ratio"] if "Future_close_ratio" in predictor.data.columns else ["Datetime", "Open", "High", "Low", "Close", "Volume"]
        st.dataframe(
            predictor.data[display_cols].tail(20).reset_index(drop=True),
            use_container_width=True,
        )

else:
    st.info("Configure parameters in the sidebar and click **Run Prediction**.")
