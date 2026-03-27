import streamlit as st
from leveledge import Predictor
from leveledge.constants import ALLOWED_INTERVALS, US_EASTERN

@st.dialog("Prediction Result", width="medium")
def show_prediction_dialog(ticker, prediction, price_level, tgt_datetime, metrics, candles_ahead=None):
    st.write(
        f'### {ticker.upper()} has a ***{prediction:.2%}*** chance of being above '
        f'${price_level:.2f} at {tgt_datetime.strftime("%m/%d/%Y %I:%M %p")}'
    )
    st.write(f'### Model Metrics')
    st.write(f'AUC: {metrics[0]:.4f}')
    st.write(f'PS: {metrics[1]:.4f}')
    st.write(f'PR: {metrics[2]:.4f}')
    st.write(f'Candles Ahead: {candles_ahead}')

st.title('LevelEdge — Predictor')

with st.form('predict_form'):
    ticker = st.text_input('Ticker (e.g. SPY, ETH-USD)', value='SPY')
    raw_tgt_datetime = st.datetime_input('Target datetime (US/Eastern)')
    if raw_tgt_datetime.tzinfo is None:
        tgt_datetime = raw_tgt_datetime.replace(tzinfo=US_EASTERN)
    else:
        tgt_datetime = raw_tgt_datetime.astimezone(US_EASTERN)
    intvl = st.selectbox('Interval', options=ALLOWED_INTERVALS, index=ALLOWED_INTERVALS.index('15m'))
    price_level = st.number_input('Price level ($)', value=0.0, format='%.2f')
    submit = st.form_submit_button('Run prediction')

if submit:
    try:
        predictor = Predictor(ticker, tgt_datetime, intvl, float(price_level))

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

        show_prediction_dialog(
            ticker,
            prediction,
            price_level,
            tgt_datetime,
            predictor.get_xgb_model_metrics(),
            predictor.candles_ahead,
        )

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
