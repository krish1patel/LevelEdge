from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

import pandas as pd
import streamlit as st
import yfinance as yf

from leveledge import Predictor
from leveledge.constants import ALLOWED_INTERVALS, EASTERN_TZ


st.set_page_config(page_title="Leveledge Strike Explorer", layout="wide")
st.title("Leveledge Option Strike Explorer")
st.write(
    "Use Leveledge predictions to scout option strikes priced around your preferred moneyness. "
    "Pick a ticker + interval, choose ITM/ATM/OTM, and we will fetch the nearest expirations, "
    "evaluate each strike with the Predictor, and tell you whether a call or put setup makes more sense."
)


def next_market_close() -> datetime:
    now = datetime.now(EASTERN_TZ)
    close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    if now >= close:
        close += timedelta(days=1)
    while close.weekday() >= 5:  # skip weekends
        close += timedelta(days=1)
    return close


def pick_strikes(strikes: List[float], last_price: float, mode: str, count: int) -> List[float]:
    strikes = sorted(strikes)
    if mode == "ATM":
        ordered = sorted(strikes, key=lambda s: abs(s - last_price))
    elif mode == "OTM":
        ordered = [s for s in strikes if s > last_price]
    else:  # ITM
        ordered = [s for s in strikes if s < last_price]
        ordered.reverse()
    if not ordered:
        ordered = strikes
    return ordered[:count]


def run_prediction(ticker: str, interval: str, strike: float, target_dt: datetime):
    predictor = Predictor(ticker, target_dt, interval, strike)
    predictor.train_xgb()
    probability = predictor.predict_xgb()
    metrics = predictor.get_xgb_model_metrics()
    contract_type = "Call" if probability >= 0.5 else "Put"
    return probability, metrics, contract_type


with st.sidebar:
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    interval = st.selectbox("Interval", options=ALLOWED_INTERVALS, index=ALLOWED_INTERVALS.index("30m"))
    strike_mode = st.selectbox("Strike style", ["ATM", "ITM", "OTM"], index=0)
    strike_count = st.slider("Number of strikes", min_value=1, max_value=10, value=3)
    expiry_choice = st.selectbox("Expiry preference", ["Nearest", "Second-nearest"], index=0)
    run_btn = st.button("Run predictions", type="primary")

if run_btn:
    if not ticker:
        st.error("Please enter a ticker symbol.")
    else:
        try:
            tk = yf.Ticker(ticker)
            expirations = tk.options
        except Exception as exc:
            st.error(f"Unable to fetch option data: {exc}")
            st.stop()

        if not expirations:
            st.warning("No option expirations found for this ticker.")
            st.stop()

        exp_index = 0 if expiry_choice == "Nearest" else min(1, len(expirations) - 1)
        expiry = expirations[exp_index]

        try:
            chain = tk.option_chain(expiry)
            last_quote = tk.history(period="1d").tail(1)
            last_price = float(last_quote["Close"].iloc[-1])
        except Exception as exc:
            st.error(f"Failed to load option chain: {exc}")
            st.stop()

        target_dt = next_market_close()
        st.subheader(f"Using expiration {expiry} · Target {target_dt.strftime('%Y-%m-%d %H:%M %Z')}")

        strikes = chain.calls["strike"].tolist()
        candidate_strikes = pick_strikes(strikes, last_price, strike_mode, strike_count)
        if not candidate_strikes:
            st.warning("No strikes matched the criteria.")
            st.stop()

        rows = []
        for strike in candidate_strikes:
            with st.spinner(f"Evaluating strike {strike:.2f} ..."):
                try:
                    probability, metrics, contract_type = run_prediction(
                        ticker, interval, strike, target_dt
                    )
                except Exception as exc:
                    st.warning(f"Prediction failed for strike {strike:.2f}: {exc}")
                    continue
                rows.append(
                    {
                        "Strike": strike,
                        "Suggested leg": contract_type,
                        "Probability (above strike)": probability,
                        "Model AUC": metrics[0],
                        "Model Precision": metrics[1],
                        "Model PR": metrics[2],
                    }
                )

        if rows:
            df = pd.DataFrame(rows)
            df["Probability (above strike)"] = df["Probability (above strike)"].apply(lambda v: f"{v:.2%}")
            df["Model AUC"] = df["Model AUC"].apply(lambda v: f"{v:.3f}")
            df["Model Precision"] = df["Model Precision"].apply(lambda v: f"{v:.3f}")
            df["Model PR"] = df["Model PR"].apply(lambda v: f"{v:.3f}")
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No predictions were generated.")
else:
    st.info("Enter a ticker and click 'Run predictions' to evaluate strike levels.")
