"""
Intraday Price Path — Streamlit UI
===================================
An interactive Streamlit app for visualizing LevelEdge probability surfaces
across intraday price levels and target times.

Run:
    streamlit run examples/intraday_price_path.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from typing import Literal

from leveledge import Predictor
from leveledge.constants import US_EASTERN, ALLOWED_INTERVALS

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intraday Price Path — LevelEdge",
    page_icon="📈",
    layout="wide",
)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def parse_interval_min(interval: str) -> int:
    if 'm' in interval:
        return int(interval[:-1])
    if 'h' in interval:
        return 60 * int(interval[:-1])
    if 'd' in interval:
        return 60 * 24 * int(interval[:-1])
    return 5


def get_market_day_times(ticker_str: str, interval: str):
    """Returns (market_open, market_close, list[target_dt]) for today.

    Target times are filtered to be at least (interval_min + 1) minutes
    from the current candle to avoid candles_ahead <= 0 errors.
    """
    from datetime import time as dt_time
    is_crypto = '-' in ticker_str.upper()
    now = datetime.now(tz=US_EASTERN)
    today_date = now.date()
    interval_min = parse_interval_min(interval)
    min_buffer = timedelta(minutes=interval_min + 1)  # target must be this far ahead

    if is_crypto:
        market_open = now.replace(hour=0, minute=0, second=0, microsecond=0)
        market_close = market_open + timedelta(days=1)
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        target_times = [
            t for i in range(24)
            if (t := next_hour + timedelta(hours=i)) <= market_close
            and t - now >= min_buffer
        ]
        return market_open, market_close, target_times

    market_open = datetime.combine(today_date, dt_time(9, 30), tzinfo=US_EASTERN)
    market_close = datetime.combine(today_date, dt_time(16, 0), tzinfo=US_EASTERN)

    if now < market_open:
        next_point = market_open
    elif now > market_close:
        next_point = market_open
        market_close = market_open + timedelta(days=1)
    else:
        next_point = now.replace(second=0, microsecond=0) + min_buffer

    target_times = []
    while next_point < market_close:
        target_times.append(next_point)
        next_point += timedelta(minutes=interval_min)

    return market_open, market_close, target_times


def build_price_levels(current_price: float) -> list[float]:
    """Symmetric percentage deviations around current price."""
    pct_devs = [-0.05, -0.03, -0.02, -0.01, -0.005, 0.005, 0.01, 0.02, 0.03, 0.05]
    levels = [current_price * (1 + p) for p in pct_devs]
    return sorted({round(l / 0.25) * 0.25 for l in levels})


def run_predictions(
    ticker: str,
    interval: str,
    evaluate: bool,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Core prediction loop. Optionally reports progress via callback(idx, total, message).
    """
    _, _, target_times = get_market_day_times(ticker, interval)

    # Get current price from a single dummy Predictor
    dummy = Predictor(ticker, target_times[0], interval, 0.0)
    current_price = dummy.current_price
    price_levels = build_price_levels(current_price)

    results: list[dict] = []
    total = len(target_times) * len(price_levels)

    for i, target_dt in enumerate(target_times):
        for j, price_level in enumerate(price_levels):
            idx = i * len(price_levels) + j + 1
            pct = (price_level / current_price - 1) * 100

            try:
                p = Predictor(ticker, target_dt, interval, price_level)
                p.train_xgb(evaluate=evaluate)
                prob = p.predict_xgb()
                candles = p.candles_ahead
            except ValueError as e:
                # candles_ahead too small for this target — skip
                prob = np.nan
                candles = np.nan
            except Exception:
                prob = np.nan
                candles = np.nan

            results.append({
                'target_time': target_dt,
                'target_hour': target_dt.hour + target_dt.minute / 60,
                'target_label': target_dt.strftime('%I:%M %p'),
                'price_level': price_level,
                'price_pct': pct,
                'probability': prob,
                'candles_ahead': candles,
                'current_price': current_price,
            })

            if progress_callback:
                progress_callback(
                    idx, total,
                    f"{target_dt.strftime('%I:%M %p')} @ ${price_level:.2f} ({pct:+.2f}%)"
                )

    return pd.DataFrame(results)


def compute_breakeven(df: pd.DataFrame) -> pd.DataFrame:
    """Linearly interpolate the price level where P=50% at each target time."""
    rows = []
    for hour, grp in df.groupby('target_hour'):
        grp = grp.dropna(subset=['probability']).sort_values('price_pct')
        if len(grp) < 2:
            continue
        below = grp[grp['probability'] < 0.5]
        above = grp[grp['probability'] >= 0.5]
        if below.empty or above.empty:
            continue
        p0 = below.iloc[-1]
        p1 = above.iloc[0]
        frac = (0.5 - p0['probability']) / (p1['probability'] - p0['probability'])
        breakeven_pct = p0['price_pct'] + frac * (p1['price_pct'] - p0['price_pct'])
        breakeven_price = p0['current_price'] * (1 + breakeven_pct / 100)
        rows.append({
            'Time': p0['target_label'],
            'Hour': hour,
            'Breakeven ($)': round(breakeven_price, 2),
            'Δ from current': f"{breakeven_pct:+.2f}%",
        })
    return pd.DataFrame(rows).sort_values('Hour')


def pivot_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return a (price_pct × target_hour) probability matrix."""
    return df.pivot_table(
        index='price_pct',
        columns='target_hour',
        values='probability',
        aggfunc='mean'
    )


# ─── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Settings")

ticker_input = st.sidebar.text_input(
    "Ticker",
    value="SPY",
    help="Stock ticker (e.g. SPY, AAPL) or crypto pair (e.g. BTC-USD)",
).strip().upper()

interval_options = [iv for iv in ALLOWED_INTERVALS if iv in ('1m', '2m', '5m', '15m', '30m', '1h', '90m')]
interval = st.sidebar.selectbox(
    "Interval",
    options=interval_options,
    index=2,  # default 5m
    help="Candle interval used for fetching data and stepping through the day",
)

evaluate = st.sidebar.checkbox(
    "Run walk-forward CV",
    value=False,
    help="Compute AUC / PS / PR metrics during training. "
         "This is much slower — disable for quick exploration.",
)

st.sidebar.divider()
st.sidebar.caption(
    "Price levels are auto-generated ±5% around current price. "
    "Target times span the market day (9:30 AM – 4:00 PM ET for stocks, "
    "24 hourly points for crypto)."
)

# ─── Main page ────────────────────────────────────────────────────────────────

st.title("📈 Intraday Price Path")
st.caption(
    "LevelEdge predictions across multiple price levels and target times "
    "throughout the current market day."
)

# Placeholder that will be replaced once we have a dummy predictor
ticker_placeholder = st.empty()

# ── Run button ───────────────────────────────────────────────────────────────
run = st.button(
    "🚀 Run Predictions",
    type="primary",
    use_container_width=True,
)

# ── Quick info banner (shown after first run) ────────────────────────────────
if run or "results_cached" in st.session_state:
    if run:
        # Reset cache flag on new run
        for key in list(st.session_state.keys()):
            if key.startswith("results_cached") or key == "df":
                del st.session_state[key]
        st.session_state["results_cached"] = True

    if "df" not in st.session_state:
        try:
            _, _, target_times = get_market_day_times(ticker_input, interval)
            dummy = Predictor(ticker_input, target_times[0], interval, 0.0)
            current_price = dummy.current_price
            price_levels = build_price_levels(current_price)
            n_targets = len(target_times)
            n_levels = len(price_levels)
            total_preds = n_targets * n_levels

            # Progress UI outside the prediction loop
            progress_bar = st.progress(0.0, text="Initializing…")
            status_text = st.empty()

            def on_progress(idx: int, total: int, msg: str):
                progress_bar.progress(idx / total)
                status_text.caption(f"Running {idx}/{total} — {msg}")

            df = run_predictions(
                ticker=ticker_input,
                interval=interval,
                evaluate=evaluate,
                progress_callback=on_progress,
            )

            progress_bar.empty()
            status_text.empty()
            st.session_state["df"] = df

        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    df = st.session_state["df"]
    _, _, target_times = get_market_day_times(ticker_input, interval)
    dummy = Predictor(ticker_input, target_times[0], interval, 0.0)
    current_price = dummy.current_price
    price_levels = build_price_levels(current_price)
    n_targets = len(target_times)
    n_levels = len(price_levels)
    total_preds = n_targets * n_levels

    # ── Info banner ──────────────────────────────────────────────────────────
    is_crypto = '-' in ticker_input.upper()
    market_label = "Crypto (24h)" if is_crypto else "NYSE (9:30 AM – 4:00 PM ET)"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${df['current_price'].iloc[0]:.2f}")
    col2.metric("Price Levels", str(len(price_levels)))
    col3.metric("Target Times", str(n_targets))
    col4.metric("Total Predictions", str(total_preds))

    st.caption(f"Market: **{market_label}** | Interval: **{interval}** | "
               f"Walk-forward CV: **{'On' if evaluate else 'Off'}**")

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_heatmap, tab_timeseries, tab_breakeven, tab_data = st.tabs(
        ["🔥 Heatmap", "📉 Time Series", "⚖️ Breakeven", "📋 Raw Data"]
    )

    with tab_heatmap:
        st.subheader("Probability Heatmap")

        pivot = pivot_matrix(df)
        pivot_reset = pivot.reset_index().melt(
            id_vars='price_pct',
            var_name='target_hour',
            value_name='probability'
        )
        pivot_reset['price_label'] = pivot_reset['price_pct'].apply(
            lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"
        )
        pivot_reset['time_label'] = pivot_reset['target_hour'].apply(
            lambda h: f"{int(h)}:{int((h % 1) * 60):02d}"
        )

        fig_heat = px.imshow(
            pivot.values,
            x=[f"{h:.1f}" for h in pivot.columns],
            y=[f"{p:+.2f}%" for p in pivot.index],
            color_continuous_scale='RdYlGn',
            range_color=[0, 1],
            labels={'x': 'Hour (ET)', 'y': 'Price Level', 'color': 'P(price > level)'},
            aspect='auto',
        )
        fig_heat.update_layout(
            title=dict(
                text=f"{ticker_input} — P(price > level) by time & price",
                x=0.5,
                font=dict(size=16),
            ),
            xaxis_title="Hour of Day (ET)",
            yaxis_title="Price Level (% from current)",
            coloraxis_colorbar_title="Probability",
            font=dict(size=11),
            margin=dict(l=80, r=40, t=80, b=60),
        )
        if 0.0 in pivot.index:
            fig_heat.add_hline(
                y=0, line_color='white', line_width=1.5, line_dash='dot',
                annotation_text="Current price", annotation_position="top left"
            )
        if not is_crypto and 9.5 in pivot.columns:
            fig_heat.add_vline(
                x=9.5,
                line_color='cyan', line_width=1, line_dash='dot',
                annotation_text="Open", annotation_position="top right"
            )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.info(
            "🟩 Green = high probability the price will be **above** that level. "
            "🟥 Red = high probability the price will be **below** that level. "
            "White dashed line = current price."
        )

    with tab_timeseries:
        st.subheader("Time Series at Each Price Level")

        level_opts = sorted(df['price_pct'].unique())
        # Default: current-price level (if present), first, and last level
        default_pcts = []
        if 0.0 in level_opts:
            default_pcts.append(0.0)
        if level_opts:
            default_pcts.extend([level_opts[0], level_opts[-1]])
        selected_levels = st.multiselect(
            "Select price levels to compare",
            options=level_opts,
            default=default_pcts,
            format_func=lambda p: f"{p:+.2f}% (${df[df['price_pct'] == p]['price_level'].iloc[0]:.2f})",
        )

        fig_line = go.Figure()
        colors = px.colors.qualitative.Plotly

        for k, pct in enumerate(selected_levels):
            slice_df = df[df['price_pct'] == pct].sort_values('target_hour').dropna(subset=['probability'])
            fig_line.add_trace(go.Scatter(
                x=slice_df['target_hour'],
                y=slice_df['probability'],
                mode='lines+markers',
                name=f"{pct:+.2f}%",
                line=dict(color=colors[k % len(colors)], width=2),
                marker=dict(size=5),
            ))

        fig_line.add_hline(
            y=0.5, line_color='gray', line_width=1, line_dash='dash',
            annotation_text="50% threshold"
        )
        fig_line.update_layout(
            title=dict(text=f"{ticker_input} — Probability Over Time", x=0.5, font=dict(size=16)),
            xaxis_title="Hour of Day (ET)",
            yaxis_title="Probability price > level",
            yaxis_range=[0, 1],
            legend_title="Price Level",
            font=dict(size=11),
            margin=dict(l=60, r=40, t=80, b=60),
            hovermode='x unified',
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with tab_breakeven:
        st.subheader("Breakeven Price Levels (P = 50%)")

        be = compute_breakeven(df)

        if be.empty:
            st.warning(
                "Could not compute breakeven — not enough price levels straddle the 50% "
                "probability threshold at any target time. "
                "Try adding more extreme price levels or using a different interval."
            )
        else:
            fig_be = go.Figure()
            fig_be.add_trace(go.Scatter(
                x=be['Hour'],
                y=be['Breakeven ($)'],
                mode='lines+markers',
                name='Breakeven Price',
                line=dict(color='crimson', width=2.5),
                marker=dict(size=8, symbol='diamond'),
            ))
            fig_be.add_hline(
                y=current_price,
                line_color='blue',
                line_width=1.5,
                line_dash='dot',
                annotation_text=f"Current ${current_price:.2f}",
                annotation_position="top right",
            )
            fig_be.update_layout(
                title=dict(text=f"{ticker_input} — Breakeven Price Level Over Time", x=0.5, font=dict(size=16)),
                xaxis_title="Hour of Day (ET)",
                yaxis_title="Price ($)",
                font=dict(size=11),
                margin=dict(l=60, r=40, t=80, b=60),
                hovermode='x unified',
            )
            st.plotly_chart(fig_be, use_container_width=True)

            st.dataframe(
                be[['Time', 'Breakeven ($)', 'Δ from current']].reset_index(drop=True),
                hide_index=True,
                use_container_width=True,
            )

            # % change from current
            be_pct = (be['Breakeven ($)'] / current_price - 1) * 100
            st.caption(
                f"Breakeven price ranges from "
                f"**${be['Breakeven ($)'].min():.2f}** ({be_pct.min():+.2f}%) to "
                f"**${be['Breakeven ($)'].max():.2f}** ({be_pct.max():+.2f}%) "
                f"relative to current price."
            )

    with tab_data:
        st.subheader("Raw Prediction Data")

        download_df = df.copy()
        download_df['target_time'] = download_df['target_time'].astype(str)
        download_csv = download_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download CSV",
            data=download_csv,
            file_name=f"{ticker_input}_intraday_predictions.csv",
            mime='text/csv',
            use_container_width=True,
        )

        display_df = df.copy()
        display_df['probability'] = display_df['probability'].map(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )
        display_df['target_time'] = display_df['target_time'].apply(
            lambda dt: dt.strftime('%I:%M %p') if hasattr(dt, 'strftime') else str(dt)
        )
        display_df = display_df.sort_values(['target_hour', 'price_pct'])
        st.dataframe(
            display_df[['target_time', 'price_level', 'price_pct', 'probability', 'candles_ahead']],
            hide_index=True,
            use_container_width=True,
        )

else:
    # Pre-run state — show placeholder description
    st.info(
        "👆 Configure ticker and interval in the **sidebar**, then click "
        "**Run Predictions** to generate the price path visualization."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        ### 🔥 Probability Heatmap
        Color-coded grid showing probability of price being **above** each level
        at every combination of time and price. Green = bullish, red = bearish.
        """)
    with col_b:
        st.markdown("""
        ### 📉 Time Series
        Compare probability curves for multiple price levels over the course
        of the day. See how bullish/bearish bias shifts as targets get closer.
        """)
    st.markdown("""
    ### ⚖️ Breakeven Table
    Interpolates the exact price level where probability = 50% at each target time —
    the market's implied breakeven for that prediction window.
    """)