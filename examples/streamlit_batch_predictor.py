import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from leveledge import Predictor
from leveledge.constants import ALLOWED_INTERVALS, US_EASTERN


def _parse_price_levels(
    raw: str,
    current_price: float | None,
    is_percentage_mode: bool,
) -> list[float]:
    """Parse comma-separated price level strings, converting percentages to absolute prices."""
    price_levels = []
    for p in raw.split(","):
        p = p.strip()
        if not p:
            continue
        if is_percentage_mode:
            if p.endswith("%"):
                try:
                    pct = float(p[:-1])
                except ValueError:
                    raise ValueError(f"Invalid percentage value: {p}")
                if current_price is None:
                    raise ValueError(
                        "Percentage mode requires a current price, but could not fetch current price for this ticker."
                    )
                price_levels.append(current_price * (1 + pct / 100))
            else:
                raise ValueError(
                    f"In percentage mode, all price levels must end with '%': got '{p}'"
                )
        else:
            try:
                price_levels.append(float(p))
            except ValueError:
                raise ValueError(f"Invalid price level: {p}")
    return price_levels


def perc_pct_str(pct: float) -> str:
    """Format a percentage change with sign, e.g. 5.0 → '+5.00%', -2.3 → '-2.30%'."""
    return f"{pct:+.2f}%"


def _get_current_price(ticker: str, end_datetime: datetime | None) -> float:
    """Fetch the most recent close price for a ticker via yfinance."""
    t = yf.ticker.Ticker(ticker)
    if end_datetime is not None:
        start = end_datetime - timedelta(days=7)
        df = t.history(start=start, end=end_datetime, interval="1m", prepost=False)
    else:
        df = t.history(period="5d", interval="1m", prepost=False)
    if df.empty:
        raise ValueError(f"Could not fetch current price for {ticker}.")
    return float(df["Close"].iloc[-1])


def run_batch_prediction(
    ticker: str,
    tgt_datetime: datetime,
    intervals: list[str],
    price_levels: list[float],
    end_datetime: datetime | None = None,
):
    all_results = []
    current_price = None

    for price_level in price_levels:
        results_for_price = []
        for intvl in intervals:
            predictor = Predictor(
                ticker, tgt_datetime, intvl, float(price_level), end_datetime=end_datetime
            )
            predictor.train_xgb()
            prediction = predictor.predict_xgb()

            if current_price is None:
                current_price = getattr(predictor, "current_price", None)

            results_for_price.append(
                {
                    "interval": intvl,
                    "prediction": prediction,
                    "metrics": predictor.get_xgb_model_metrics(),
                    "candles_ahead": predictor.candles_ahead,
                }
            )

        all_results.append({"price_level": price_level, "results": results_for_price})

    return all_results, current_price


@st.dialog("Batch Prediction Results", width="large")
def show_batch_prediction_dialog(
    ticker,
    tgt_datetime,
    all_results,
    use_interval_preset=False,
    current_price=None,
    is_backtest=False,
    end_datetime=None,
):
    if not all_results:
        st.warning("No results to display.")
        return

    if is_backtest and end_datetime is not None:
        st.info(
            f"**Backtest** — Data as of {end_datetime.strftime('%m/%d/%Y %I:%M %p')} (US/Eastern)"
        )

    if current_price is not None:
        st.markdown(
            f"### Current {ticker.upper()} price: **${current_price:.2f}**"
        )

    for block in all_results:
        price_level = block["price_level"]
        results = block["results"]

        if current_price is not None and current_price != 0:
            pct_move = ((price_level - current_price) / current_price) * 100
            price_label = f"${price_level:.2f} ({perc_pct_str(pct_move)})"
        else:
            price_label = f"${price_level:.2f}"

        st.write(
            f'### {ticker.upper()} predictions for price level '
            f'{price_label} at {tgt_datetime.strftime("%m/%d/%Y %I:%M %p")}'
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
            st.warning("No results to display for this price level.")
            continue

        df = pd.DataFrame(rows)

        order_map = {iv: i for i, iv in enumerate(ALLOWED_INTERVALS)}
        df["__sort_key"] = df["Interval"].map(order_map)
        df = df.sort_values("__sort_key").drop(columns="__sort_key").reset_index(drop=True)

        if use_interval_preset:
            # Weighted moving average (WMA) row for preset intervals 15m, 30m, 90m
            weight_map = {"15m": 0.45, "30m": 0.40, "90m": 0.15}
            preset_df = df[df["Interval"].isin(weight_map.keys())].copy()

            if not preset_df.empty:
                weights = preset_df["Interval"].map(weight_map)
                total_weight = weights.sum()

                if total_weight > 0:
                    wma_row = {
                        "Interval": "WMA (15m/30m/90m)",
                        "Prediction %": (preset_df["Prediction %"] * weights).sum() / total_weight,
                        "AUC": (preset_df["AUC"] * weights).sum() / total_weight,
                        "PS": (preset_df["PS"] * weights).sum() / total_weight,
                        "PR": (preset_df["PR"] * weights).sum() / total_weight,
                        "Candles Ahead": (preset_df["Candles Ahead"] * weights).sum() / total_weight,
                    }
                    df = pd.concat([df, pd.DataFrame([wma_row])], ignore_index=True)
        else:
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

        # Always-visible core columns (no index)
        core_cols = ["Interval", "Prediction %"]
        st.dataframe(
            df_display[core_cols],
            hide_index=True,
            width="stretch",
        )

        # Foldable detailed metrics (no index)
        metrics_cols = ["Interval", "AUC", "PS", "PR", "Candles Ahead"]
        with st.expander("Metrics & candles ahead", expanded=False):
            st.dataframe(
                df_display[metrics_cols],
                hide_index=True,
                width="stretch",
            )


st.title("LevelEdge — Batch Predictor")

# Put the preset checkbox outside the form so that toggling it
# immediately triggers a rerun and updates the disabled state below.
use_interval_preset = st.checkbox(
    "Use 15m/30m/90m interval preset",
    value=True,
    help="When checked, predictions will run for 15m, 30m, and 90m intervals using a weighted moving average.",
)

backtest_mode = st.checkbox(
    "Enable backtest mode",
    value=False,
    help="Use historical data up to an end datetime instead of live data. Target datetime must be after the end datetime.",
)

with st.form("batch_predict_form"):
    ticker = st.text_input("Ticker (e.g. SPY, ETH-USD)", value="SPY")
    st.set_page_config(page_title=f'Predicting {ticker.upper()}')
    raw_tgt_datetime = st.datetime_input("Target datetime (US/Eastern)")
    if raw_tgt_datetime.tzinfo is None:
        tgt_datetime = raw_tgt_datetime.replace(tzinfo=US_EASTERN)
    else:
        tgt_datetime = raw_tgt_datetime.astimezone(US_EASTERN)

    end_datetime = None
    if backtest_mode:
        raw_end_datetime = st.datetime_input(
            "End datetime for stock data (US/Eastern)",
            help="Historical data is fetched up to this time. Target datetime must be after this.",
        )
        if raw_end_datetime.tzinfo is None:
            end_datetime = raw_end_datetime.replace(tzinfo=US_EASTERN)
        else:
            end_datetime = raw_end_datetime.astimezone(US_EASTERN)

    intervals = st.multiselect(
        "Intervals",
        options=ALLOWED_INTERVALS,
        default=["15m"],
        help="Select one or more intervals to run predictions for.",
        disabled=use_interval_preset,
    )

    if use_interval_preset:
        # Force preset intervals when the preset is enabled
        intervals = ["15m", "30m", "90m"]

    price_mode_pct = st.checkbox(
        "Use percentage mode",
        value=False,
        help="When checked, price levels are treated as percentage moves from the current price (e.g. '+2%', '-1.5%'). The '%' suffix is required on each entry.",
    )

    if price_mode_pct:
        label = "Price levels (% move, comma-separated)"
        placeholder = "+2%, -1%, +5%"
        default = "+2%, -2%"
    else:
        label = "Price levels ($, comma-separated)"
        placeholder = "400, 405.5"
        default = "0.0"

    price_levels_raw = st.text_input(label, value=default, placeholder=placeholder)
    submit = st.form_submit_button("Run batch prediction")

if submit:
    validation_failed = False

    if backtest_mode:
        if end_datetime is None:
            st.error("Backtest mode requires an end datetime for stock data.")
            validation_failed = True
        elif tgt_datetime <= end_datetime:
            st.error(
                "In backtest mode, target datetime must be after the end datetime for stock data."
            )
            validation_failed = True
    else:
        # Prevent running with an invalid / past target datetime so we never
        # open a dialog that can only error.
        if tgt_datetime < datetime.now(tz=US_EASTERN):
            st.error("Target datetime must be in the future (US/Eastern).")
            validation_failed = True

    if not intervals:
        st.error("Please select at least one interval.")
        validation_failed = True

    price_levels: list[float] = []
    if not validation_failed:
        try:
            if price_mode_pct:
                current_price_for_pct = _get_current_price(ticker, end_datetime if backtest_mode else None)
                price_levels = _parse_price_levels(
                    price_levels_raw, current_price_for_pct, is_percentage_mode=True
                )
            else:
                price_levels = _parse_price_levels(
                    price_levels_raw, current_price=None, is_percentage_mode=False
                )
        except ValueError as e:
            st.error(str(e))
            validation_failed = True

    if not validation_failed and not price_levels:
        st.error("Please enter at least one valid price level.")
        validation_failed = True

    if not validation_failed:
        try:
            with st.spinner("Running batch predictions..."):
                all_results, current_price = run_batch_prediction(
                    ticker=ticker,
                    tgt_datetime=tgt_datetime,
                    intervals=intervals,
                    price_levels=price_levels,
                    end_datetime=end_datetime if backtest_mode else None,
                )
            show_batch_prediction_dialog(
                ticker,
                tgt_datetime,
                all_results,
                use_interval_preset=use_interval_preset,
                current_price=current_price,
                is_backtest=backtest_mode,
                end_datetime=end_datetime if backtest_mode else None,
            )
        except Exception as e:
            st.error(f"Error running batch predictions: {e}")

