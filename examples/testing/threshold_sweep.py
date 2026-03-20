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

df = df.dropna(subset=["outcome_price"])
df['actual_hit'] = (df['outcome_price'] > df['price_level']).astype(int)

thresholds = np.arange(0.50, 0.95, 0.05)
MIN_SAMPLE = 20
results = []

for interval, group in df[df["interval"] != "90m"].groupby("interval"):
    group = group.dropna(subset=["prediction", "actual_hit"])
    for threshold in thresholds:
        bull_mask   = group["prediction"] >= threshold
        bull_subset = group[bull_mask]
        bull_n      = len(bull_subset)
        bull_precision = bull_subset["actual_hit"].mean() if bull_n > 0 else None

        bear_mask   = group["prediction"] <= (1 - threshold)
        bear_subset = group[bear_mask]
        bear_n      = len(bear_subset)
        bear_precision = (1 - bear_subset["actual_hit"]).mean() if bear_n > 0 else None

        results.append({
            "interval":         interval,
            "threshold":        threshold,
            "bull_precision":   bull_precision,
            "bear_precision":   bear_precision,
            "bull_n":           bull_n,
            "bear_n":           bear_n,
            "bull_low_sample":  bull_n < MIN_SAMPLE,
            "bear_low_sample":  bear_n < MIN_SAMPLE,
        })

results_df = pd.DataFrame(results)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for interval, group in results_df.groupby("interval"):
    color = None

    # Bull precision
    ax = axes[0][0]
    line, = ax.plot(group["threshold"], group["bull_precision"], marker='o', label=interval)
    color = line.get_color()
    low = group[group["bull_low_sample"]]
    if not low.empty:
        ax.scatter(low["threshold"], low["bull_precision"], 
                   marker='X', s=120, color=color, zorder=5, edgecolors='red', linewidths=1.5)

    # Bull n signals
    ax = axes[0][1]
    ax.plot(group["threshold"], group["bull_n"], marker='o', label=interval, color=color)
    ax.axhline(y=MIN_SAMPLE, color='r', linestyle=':', alpha=0.5, linewidth=1)

    # Bear precision
    ax = axes[1][0]
    ax.plot(group["threshold"], group["bear_precision"], marker='o', label=interval, color=color)
    low = group[group["bear_low_sample"]]
    if not low.empty:
        ax.scatter(low["threshold"], low["bear_precision"],
                   marker='X', s=120, color=color, zorder=5, edgecolors='red', linewidths=1.5)

    # Bear n signals
    ax = axes[1][1]
    ax.plot(group["threshold"], group["bear_n"], marker='o', label=interval, color=color)
    ax.axhline(y=MIN_SAMPLE, color='r', linestyle=':', alpha=0.5, linewidth=1)

for ax in [axes[0][0], axes[1][0]]:
    ax.axhline(y=0.75, color='r', linestyle='--', label="75% target")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=7)

for ax in [axes[0][1], axes[1][1]]:
    ax.set_xlabel("Threshold")
    ax.set_ylabel("N Signals")
    ax.legend(fontsize=7)

axes[0][0].set_title("Bull Precision by Threshold  [✕ = n < 20]")
axes[0][1].set_title("Bull N Signals by Threshold  [--- = min sample]")
axes[1][0].set_title("Bear Precision by Threshold  [✕ = n < 20]")
axes[1][1].set_title("Bear N Signals by Threshold  [--- = min sample]")

plt.tight_layout()
plt.show()
