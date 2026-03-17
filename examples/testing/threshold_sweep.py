import pandas as pd
from supabase import create_client
import os
from dotenv import load_dotenv

import numpy as np
import matplotlib.pyplot as plt

load_dotenv()
supabase = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])

table_name = 'logs'

table_length = supabase.table(table_name).select('*', count='exact', head=True).execute().count

df = pd.DataFrame()

if table_length > 1000:
    for index in range(0, table_length, 1000):
        df = pd.concat([df, pd.DataFrame(supabase.table(table_name).select("*").range(index, index+999).execute().data)])
else:
    df = pd.DataFrame(supabase.table(table_name).select('*').execute().data)

df['actual_hit'] = (df['outcome_price'] > df['price_level']).astype(int)



thresholds = np.arange(0.50, 0.95, 0.05)
results = []

for interval, group in df[df["interval"] != "90m"].groupby("interval"):
    group = group.dropna(subset=["prediction", "actual_hit"])
    
    for threshold in thresholds:
        mask = group["prediction"] >= threshold
        subset = group[mask]
        
        if len(subset) < 20:  # ignore thresholds with too few signals
            continue
        
        precision = subset["actual_hit"].mean()
        
        results.append({
            "interval": interval,
            "threshold": threshold,
            "precision": precision,
            "n_signals": len(subset),
            "pct_of_total": len(subset) / len(group)
        })

results_df = pd.DataFrame(results)




fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for interval, group in results_df.groupby("interval"):
    axes[0].plot(group["threshold"], group["precision"], marker='o', label=interval)
    axes[1].plot(group["threshold"], group["n_signals"], marker='o', label=interval)

axes[0].axhline(y=0.75, color='r', linestyle='--', label="75% target")
axes[0].set_title("Precision by Threshold")
axes[0].set_xlabel("Threshold")
axes[0].set_ylabel("Precision")
axes[0].legend()

axes[1].set_title("Number of Signals by Threshold")
axes[1].set_xlabel("Threshold")
axes[1].set_ylabel("N Signals")
axes[1].legend()

plt.tight_layout()
plt.show()
