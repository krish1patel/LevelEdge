import pandas as pd
from supabase import create_client
import os
from dotenv import load_dotenv

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

load_dotenv()
supabase = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])

table_name = 'new_backtest_logs'

table_length = supabase.table(table_name).select('*', count='exact', head=True).execute().count

df = pd.DataFrame()

if table_length > 1000:
    for index in range(0, table_length, 1000):
        df = pd.concat([df, pd.DataFrame(supabase.table(table_name).select("*").range(index, index+999).execute().data)])
else:
    df = pd.DataFrame(supabase.table(table_name).select('*').execute().data)

df['actual_hit'] = (df['outcome_price'] > df['price_level']).astype(int)


fig, axes = plt.subplots(1, len(df["interval"].unique()), figsize=(18, 5))

n_bins = 20

for ax, interval in zip(axes, df["interval"].unique()):
    subset = df[df["interval"] == interval].dropna(subset=["prediction", "actual_hit"])
    if len(subset) < 50:
        continue
    
    prob_true, prob_pred = calibration_curve(
        subset["actual_hit"], subset["prediction"], n_bins=n_bins, strategy="quantile"
    )
    
    ax.plot(prob_pred, prob_true, marker='o', label=interval)
    ax.plot([0, 1], [0, 1], 'k--', label="Perfect")
    ax.set_title(f"{interval}")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend()

plt.tight_layout()
plt.show()