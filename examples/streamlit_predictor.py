import streamlit as st
import pandas as pd
from datetime import datetime

from leveledge import Predictor
from leveledge.constants import ALLOWED_INTERVALS, EASTERN_TZ

st.set_page_config(page_title="Leveledge streamlit predictor", layout="wide")

allowed_intervals = [i for i in ALLOWED_INTERVALS if i != "10m"]

st.title("Leveledge streamlit predictor")
st.markdown(
    "Use multiple allowed intervals to train Predictor and tabulate the resulting probabilities and metrics, including an average row."
)

with st.sidebar:
    ticker = st.text_input("Ticker", "AAPL").upper().strip()
    intervals = st.multiselect("Intervals", allowed_intervals, default=["15m"])
    target_time = st.datetime_input("Target datetime", datetime.now(EASTERN_TZ))
    price = st.number_input("Price level", value=150.0)
    run_button = st.button("Run prediction")

if run_button:
    if not ticker:
        st.error("Ticker is required")
        st.stop()
    if not intervals:
        st.error("Select at least one interval")
        st.stop()

    results = []
    for interval in intervals:
        predictor = Predictor(ticker, target_time, interval, price)
        predictor.train_xgb()
        probability = predictor.predict_xgb()
        metrics = predictor.get_xgb_model_metrics()
        results.append(
            {
                "interval": interval,
                "probability_above": probability,
                "auc": metrics[0],
                "precision": metrics[1],
                "pr": metrics[2],
            }
        )

    df = pd.DataFrame(results)
    df_display = df.copy()
    df_display["probability_above"] = df_display["probability_above"].apply(lambda v: f"{v:.2%}")
    df_display["auc"] = df_display["auc"].apply(lambda v: f"{v:.3f}")
    df_display["precision"] = df_display["precision"].apply(lambda v: f"{v:.3f}")
    df_display["pr"] = df_display["pr"].apply(lambda v: f"{v:.3f}")

    st.subheader("Results")
    st.dataframe(df_display)

    avg = {
        "interval": "Average",
        "probability_above": f"{df['probability_above'].mean():.2%}",
        "auc": f"{df['auc'].mean():.3f}",
        "precision": f"{df['precision'].mean():.3f}",
        "pr": f"{df['pr'].mean():.3f}",
    }
    st.table(pd.DataFrame([avg]))
else:
    st.info("Select intervals and run the predictor to see combined results.")
