"""Batch option predictions example

Saves a CSV of model predictions for short-dated options per user spec.

Usage:
    python examples/batch_option_predictions.py

Notes:
- Requires LevelEdge package installed (leveledge.Predictor available).
- Uses yfinance to fetch option chains.
- Writes examples/option_predictions.csv
"""
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
import yfinance as yf
import pandas as pd
import sys

# make sure package imports the local leveledge
sys.path.insert(0, '.')
from leveledge import Predictor, ensure_future_market_datetime


def next_friday_date(tz_name='US/Eastern'):
    now = datetime.now(tz=ZoneInfo('UTC')).astimezone(ZoneInfo(tz_name))
    days_ahead = (4 - now.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return (now + timedelta(days=days_ahead)).date()


def find_strikes_for_price(chain_calls, chain_puts, price_limit=0.30):
    calls = chain_calls.dropna(subset=['ask']).copy()
    puts = chain_puts.dropna(subset=['ask']).copy()
    calls_under = calls[calls['ask'] <= price_limit]
    puts_under = puts[puts['ask'] <= price_limit]
    call_strike = None
    put_strike = None
    call_ask = None
    put_ask = None
    if not calls_under.empty:
        row = calls_under.sort_values('strike').iloc[0]
        call_strike = row['strike']
        call_ask = row['ask']
    if not puts_under.empty:
        row = puts_under.sort_values('strike', ascending=False).iloc[0]
        put_strike = row['strike']
        put_ask = row['ask']
    return (call_strike, call_ask, put_strike, put_ask)


def main():
    tickers = ["AAPL","MSFT","AMZN","NVDA","TSLA","GOOG","META","NFLX","JPM","SPY"]

    target_date = next_friday_date('US/Eastern')
    raw_dt = datetime.combine(target_date, time(16,0,0))
    tgt_dt = ensure_future_market_datetime(raw_dt, tz_name='US/Eastern')

    results = []

    for tk in tickers:
        try:
            ticker = yf.Ticker(tk)
            exps = ticker.options
            if not exps:
                results.append({'ticker':tk,'error':'no options data'})
                continue
            # prefer expiry equal to target date, else nearest
            exp_str = None
            for e in exps:
                ed = datetime.strptime(e, "%Y-%m-%d").date()
                if ed == target_date:
                    exp_str = e
                    break
            if exp_str is None:
                exp_str = exps[0]
            chain = ticker.option_chain(exp_str)
            call_strike, call_ask, put_strike, put_ask = find_strikes_for_price(chain.calls, chain.puts, price_limit=0.30)

            if call_strike is None:
                results.append({'ticker':tk,'type':'call','strike':None,'ask':None,'prediction':None,'note':'no call <= $0.30'})
            else:
                predictor = Predictor(tk, tgt_dt, '1h', float(call_strike))
                predictor.train_xgb()
                pred = predictor.predict_xgb()
                results.append({'ticker':tk,'type':'call','strike':call_strike,'ask':call_ask,'prediction':pred})

            if put_strike is None:
                results.append({'ticker':tk,'type':'put','strike':None,'ask':None,'prediction':None,'note':'no put <= $0.30'})
            else:
                predictor = Predictor(tk, tgt_dt, '1h', float(put_strike))
                predictor.train_xgb()
                pred = predictor.predict_xgb()
                results.append({'ticker':tk,'type':'put','strike':put_strike,'ask':put_ask,'prediction':pred})

        except Exception as e:
            results.append({'ticker':tk,'error':str(e)})

    df = pd.DataFrame(results)
    out_path = 'examples/option_predictions.csv'
    df.to_csv(out_path, index=False)
    print(f'Wrote {out_path}')
    print(df.to_markdown(index=False))


if __name__ == '__main__':
    main()
