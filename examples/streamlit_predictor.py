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

        with st.spinner('Training model...'):
            predictor.train_xgb()

        with st.spinner('Running prediction...'):
            prediction = predictor.predict_xgb()

        show_prediction_dialog(
            ticker,
            prediction,
            price_level,
            tgt_datetime,
            predictor.get_xgb_model_metrics(),
            predictor.candles_ahead,
        )

    except Exception as e:
        st.error(f'Error running prediction: {e}')
