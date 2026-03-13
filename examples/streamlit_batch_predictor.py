import streamlit as st
import pandas as pd
import time
from datetime import datetime
from leveledge import Predictor
from leveledge.constants import ALLOWED_INTERVALS, US_EASTERN


def run_batch_prediction(ticker, tgt_datetime, intervals, price_levels):
    all_results = []
    current_price = None

    for price_level in price_levels:
        results_for_price = []
        for intvl in intervals:
            predictor = Predictor(ticker, tgt_datetime, intvl, float(price_level))
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
    ticker, tgt_datetime, all_results, use_interval_preset=False, current_price=None
):
    if not all_results:
        st.warning("No results to display.")
        return

    auto_key = "leveledge_auto_rerun_enabled"
    if auto_key not in st.session_state:
        st.session_state[auto_key] = True

    force_key = "leveledge_force_rerun"
    if force_key not in st.session_state:
        st.session_state[force_key] = False

    last_run_key = "leveledge_last_run_ts"
    if last_run_key not in st.session_state:
        st.session_state[last_run_key] = time.time()

    st.checkbox("Auto rerun every 30s", key=auto_key)

    col_l, col_r = st.columns([1, 1])
    with col_l:
        if st.button("Rerun now", use_container_width=True):
            st.session_state[force_key] = True
            st.rerun()
    with col_r:
        if st.button("Close", use_container_width=True):
            st.session_state["leveledge_show_dialog"] = False
            st.rerun()

    elapsed = time.time() - float(st.session_state[last_run_key])
    remaining = max(0, int(30 - elapsed))
    st.caption(f"Next auto rerun in **{remaining}s**" if st.session_state[auto_key] else "")

    if current_price is not None:
        st.markdown(
            f"### Current {ticker.upper()} price: **${current_price:.2f}**"
        )

    for block in all_results:
        price_level = block["price_level"]
        results = block["results"]

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

with st.form("batch_predict_form"):
    ticker = st.text_input("Ticker (e.g. SPY, ETH-USD)", value="SPY")
    st.set_page_config(page_title=f'Predicting {ticker.upper()}')
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
        disabled=use_interval_preset,
    )

    if use_interval_preset:
        # Force preset intervals when the preset is enabled
        intervals = ["15m", "30m", "90m"]

    price_levels_raw = st.text_input(
        "Price levels ($, comma-separated)",
        value="0.0",
        help="Enter one or more price levels separated by commas, e.g. 400, 405.5",
    )
    submit = st.form_submit_button("Run batch prediction")

if submit:
    if not intervals:
        st.error("Please select at least one interval.")
    else:
        try:
            price_levels = [
                float(p.strip())
                for p in price_levels_raw.split(",")
                if p.strip() != ""
            ]
        except ValueError:
            st.error(
                "Could not parse price levels. "
                "Please enter comma-separated numbers, e.g. 400, 405.5"
            )
            price_levels = []

        if not price_levels:
            st.error("Please enter at least one valid price level.")
        else:
            # Persist run parameters so auto-rerun can reuse them
            st.session_state["leveledge_last_inputs"] = {
                "ticker": ticker,
                "tgt_datetime": tgt_datetime,
                "intervals": intervals,
                "price_levels": price_levels,
                "use_interval_preset": use_interval_preset,
            }
            st.session_state["leveledge_show_dialog"] = True
            try:
                # Trigger an immediate run in the unified dialog-render block below.
                st.session_state["leveledge_force_rerun"] = True
                st.rerun()

            except Exception as e:
                st.error(f"Error running batch predictions: {e}")


# Auto-rerun loop: if dialog is open and enabled, rerun every 30s.
if st.session_state.get("leveledge_show_dialog") and st.session_state.get("leveledge_last_inputs"):
    auto_enabled = st.session_state.get("leveledge_auto_rerun_enabled", True)
    force_rerun = st.session_state.get("leveledge_force_rerun", False)
    last_run = float(st.session_state.get("leveledge_last_run_ts", 0))
    should_rerun = force_rerun or (auto_enabled and (time.time() - last_run >= 30))

    if should_rerun:
        st.session_state["leveledge_force_rerun"] = False
        inputs = st.session_state["leveledge_last_inputs"]
        try:
            with st.spinner("Auto refreshing predictions..."):
                all_results, current_price = run_batch_prediction(
                    ticker=inputs["ticker"],
                    tgt_datetime=inputs["tgt_datetime"],
                    intervals=inputs["intervals"],
                    price_levels=inputs["price_levels"],
                )
            st.session_state["leveledge_last_results"] = all_results
            st.session_state["leveledge_last_current_price"] = current_price
            st.session_state["leveledge_last_run_ts"] = time.time()
        except Exception as e:
            st.error(f"Error auto-refreshing predictions: {e}")

    # Keep dialog open across reruns by re-invoking it from state
    show_batch_prediction_dialog(
        st.session_state["leveledge_last_inputs"]["ticker"],
        st.session_state["leveledge_last_inputs"]["tgt_datetime"],
        st.session_state.get("leveledge_last_results", []),
        use_interval_preset=st.session_state["leveledge_last_inputs"].get("use_interval_preset", False),
        current_price=st.session_state.get("leveledge_last_current_price"),
    )

    # Drive a 1s ticking countdown while auto refresh is enabled
    if auto_enabled:
        time.sleep(1)
        st.rerun()

