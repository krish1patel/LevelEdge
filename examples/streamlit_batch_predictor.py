import streamlit as st
import pandas as pd

from leveledge import Predictor
from leveledge.constants import ALLOWED_INTERVALS, US_EASTERN


@st.dialog("Batch Prediction Results", width="large")
def show_batch_prediction_dialog(ticker, price_level, tgt_datetime, results):
    st.write(
        f'### {ticker.upper()} predictions for price level '
        f'${price_level:.2f} at {tgt_datetime.strftime("%m/%d/%Y %I:%M %p")}'
    )

    rows = []
    for res in results:
        auc, ps, pr = res["metrics"]
        rows.append(
            {
                "Interval": res["interval"],
                "Prediction %": res["prediction"] * 100.0,
                "AUC": auc,
                "PS": ps,
                "PR": pr,
                "Candles Ahead": res["candles_ahead"],
            }
        )

    if not rows:
        st.warning("No results to display.")
        return

    df = pd.DataFrame(rows)

    order_map = {iv: i for i, iv in enumerate(ALLOWED_INTERVALS)}
    df["__sort_key"] = df["Interval"].map(order_map)
    df = df.sort_values("__sort_key").drop(columns="__sort_key").reset_index(drop=True)

    avg_row = {
        "Interval": "Average",
        "Prediction %": df["Prediction %"].mean(),
        "AUC": df["AUC"].mean(),
        "PS": df["PS"].mean(),
        "PR": df["PR"].mean(),
        "Candles Ahead": df["Candles Ahead"].mean(),
    }
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    df_display = df.copy()
    df_display["Prediction %"] = df_display["Prediction %"].map(lambda x: f"{x:.2f}%")
    for col in ["AUC", "PS", "PR"]:
        df_display[col] = df_display[col].map(lambda x: f"{x:.4f}")
    df_display["Candles Ahead"] = df_display["Candles Ahead"].map(
        lambda x: f"{x:.0f}" if pd.notna(x) else ""
    )

    st.table(df_display)


st.title("LevelEdge — Batch Predictor")

with st.form("batch_predict_form"):
    ticker = st.text_input("Ticker (e.g. SPY, ETH-USD)", value="SPY")
    raw_tgt_datetime = st.datetime_input("Target datetime (US/Eastern)")
    if raw_tgt_datetime.tzinfo is None:
        tgt_datetime = raw_tgt_datetime.replace(tzinfo=US_EASTERN)
    else:
        tgt_datetime = raw_tgt_datetime.astimezone(US_EASTERN)

    intervals = st.multiselect(
        "Intervals",
        options=ALLOWED_INTERVALS,
        default=["15m"],
        help="Select one or more intervals to run predictions for.",
    )

    price_level = st.number_input("Price level ($)", value=0.0, format="%.2f")
    submit = st.form_submit_button("Run batch prediction")

if submit:
    if not intervals:
        st.error("Please select at least one interval.")
    else:
        results = []
        try:
            for intvl in intervals:
                with st.spinner(f"Training model for interval {intvl}..."):
                    predictor = Predictor(ticker, tgt_datetime, intvl, float(price_level))
                    predictor.train_xgb()

                with st.spinner(f"Running prediction for interval {intvl}..."):
                    prediction = predictor.predict_xgb()

                results.append(
                    {
                        "interval": intvl,
                        "prediction": prediction,
                        "metrics": predictor.get_xgb_model_metrics(),
                        "candles_ahead": predictor.candles_ahead,
                    }
                )

            show_batch_prediction_dialog(ticker, price_level, tgt_datetime, results)

        except Exception as e:
            st.error(f"Error running batch predictions: {e}")

