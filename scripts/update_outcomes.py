import os
import yfinance as yf
from supabase import create_client
from dotenv import load_dotenv
from datetime import datetime, timezone
import pandas as pd

load_dotenv()

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"]
)

def fetch_outcome_price(ticker: str, target_datetime: str, interval_minutes: int) -> float | None:
    """Fetch the closing price of the candle at or just after target_datetime."""
    try:
        target_dt = pd.Timestamp(target_datetime).tz_convert("America/New_York")
        
        # fetch a small window around the target datetime
        start = target_dt - pd.Timedelta(minutes=interval_minutes * 2)
        end = target_dt + pd.Timedelta(minutes=interval_minutes * 2)

        # pick the right yfinance interval string
        interval_map = {15: "15m", 30: "30m", 60: "1h", 90: "90m", 1440: "1d"}
        yf_interval = interval_map.get(interval_minutes, "15m")

        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(start=start, end=end, interval=yf_interval)

        if hist.empty:
            return None

        # find the candle closest to target datetime
        hist.index = hist.index.tz_convert("America/New_York")
        closest = hist.index[hist.index.get_indexer([target_dt], method="nearest")[0]]
        return float(hist.loc[closest, "Close"])

    except Exception as e:
        print(f"Error fetching price for {ticker} at {target_datetime}: {e}")
        return None


def update_outcomes(table: str):
    # fetch all rows where outcome_price is null and target_datetime has passed
    response = supabase.table(table) \
        .select("id, ticker, target_datetime, interval_minutes") \
        .is_("outcome_price", "null") \
        .lt("target_datetime", datetime.now(timezone.utc).isoformat()) \
        .execute()

    rows = response.data
    print(f"Found {len(rows)} rows to update")

    for row in rows:
        price = fetch_outcome_price(
            row["ticker"],
            row["target_datetime"],
            row["interval_minutes"]
        )

        if price is not None:
            supabase.table(table) \
                .update({"outcome_price": price}) \
                .eq("id", row["id"]) \
                .execute()
            print(f"Updated {row['ticker']} @ {row['target_datetime']} → {price}")
        else:
            print(f"Could not fetch price for {row['ticker']} @ {row['target_datetime']}")

if __name__ == "__main__":
    update_outcomes('logs')
    update_outcomes('backtest_logs')
    update_outcomes('new_backtest_logs')
    update_outcomes('forwardtest_logs')