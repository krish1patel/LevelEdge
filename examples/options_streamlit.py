from curses import raw
from pandas import options
import streamlit as st
from leveledge import Predictor
from leveledge.constants import ALLOWED_INTERVALS, US_EASTERN
import yfinance as yf

STRATEGY_OPTIONS = ['Above', 'Below', 'Around']


st.title('LevelEdge - Options Predictor')

with st.form('predict_form'):
    ticker_str = st.text_input('Ticker (e.g. SPY, NVDA)', value = 'SPY')
    ticker = yf.Ticker(ticker_str)
    options_dates = ticker.options

    raw_tgt_date = st.date_input('Expiration Date', min_value='today')
    if raw_tgt_date.tzinfo is None:
        tgt_date = raw_tgt_date.replace(tzinfo=US_EASTERN)
    else:
        tgt_date = raw_tgt_date.astimezone(US_EASTERN)

    if tgt_date.strftime("%Y-%d-%m") not in options_dates:
        st.error('No options contracts expiring on inputted date')
    interval = st.selectbox('Interval', options=ALLOWED_INTERVALS, index=ALLOWED_INTERVALS.index('2m'))
    strategy = st.selectbox("Contract Strategy (Above, Below, or Around current stock price)", options=STRATEGY_OPTIONS, index=STRATEGY_OPTIONS.index('Around'))
