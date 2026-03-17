from datetime import datetime, timedelta
from leveledge.constants import US_EASTERN
from leveledge import Predictor

end_dt = datetime(2026, 3, 7, 10, 30, tzinfo=US_EASTERN)
target_dt = datetime(2026, 3, 7, 16, 0, tzinfo=US_EASTERN)

p = Predictor("QQQ", target_dt, "2m", 480.0, end_datetime=end_dt)

print(f"Total rows in data_withna: {len(p.data_withna)}")
print(f"Rows with non-null Future_close: {p.data_withna['Future_close'].notna().sum()}")
print(f"Candles ahead: {p.candles_ahead}")
print(f"Last candle timestamp: {p.data_withna['Datetime'].iloc[-1]}")
print(f"Last Future_close: {p.data_withna['Future_close'].iloc[-1]}")
print(f"Second to last Future_close: {p.data_withna['Future_close'].iloc[-2]}")
print()
print(p.data_withna[['Datetime', 'Close', 'Future_close', 'Target']].tail(20))