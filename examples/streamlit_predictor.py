import streamlit as st
from leveledge import Predictor
from zoneinfo import ZoneInfo
from datetime import datetime


@st.dialog("Prediction Result", width="medium")
def show_prediction_dialog(ticker, prediction, price_level, tgt_datetime, metrics):
    st.write(
        f'### {ticker.upper()} has a ***{prediction:.2%}*** chance of being above '
        f'${price_level:.2f} at {tgt_datetime.strftime("%m/%d/%Y %I:%M %p")}'
    )
    st.write(f'### Model Metrics')
    st.write(f'AUC: {metrics[0]:.4f}')
    st.write(f'PS: {metrics[1]:.4f}')
    st.write(f'PR: {metrics[2]:.4f}')

st.title('LevelEdge — Predictor')

with st.form('predict_form'):
    ticker = st.text_input('Ticker (e.g. SPY, ETH-USD)', value='SPY')
    tgt_datetime = st.datetime_input('Target datetime (EST)').replace(tzinfo=ZoneInfo('EST'))
    #tgt_datetime_str = st.text_input('Target datetime (EST, YYYY-MM-DD HH:MM:SS)', value=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    intvl = st.selectbox('Interval', options=['5m','15m','30m','1h','90m'], index=1)
    price_level = st.number_input('Price level ($)', value=0.0, format='%.2f')
    submit = st.form_submit_button('Run prediction')

if submit:
    try:
        # format_string = "%Y-%m-%d %H:%M:%S"
        # tgt_datetime = datetime.strptime(tgt_datetime_str, format_string).replace(tzinfo=ZoneInfo('EST'))
        # st.write(f'Target datetime: {tgt_datetime}')
        predictor = Predictor(ticker, tgt_datetime, intvl, float(price_level))

        with st.spinner('Training model...'):
            predictor.train_xgb()

        with st.spinner('Running prediction...'):
            prediction = predictor.predict_xgb()

        # st.subheader('Model Metrics')
        # st.code(predictor.get_xgb_model_metrics(), language='json')

        show_prediction_dialog(ticker, prediction, price_level, tgt_datetime, predictor.get_xgb_model_metrics())

        # st.subheader('Prediction Result')
        # st.write(f'{ticker.upper()} has a {prediction:.2%} chance of being above ${price_level} at {tgt_datetime.strftime("%m/%d/%Y %I:%M %p")}')

    except Exception as e:
        st.error(f'Error running prediction: {e}')
