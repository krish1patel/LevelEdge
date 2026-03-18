import pandas as pd
from supabase import create_client
import os
from dotenv import load_dotenv

from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

load_dotenv()
supabase = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])

table_name = 'forwardtest_logs'

table_length = supabase.table(table_name).select('*', count='exact', head=True).execute().count

df = pd.DataFrame()

if table_length > 1000:
    for index in range(0, table_length, 1000):
        df = pd.concat([df, pd.DataFrame(supabase.table(table_name).select("*").range(index, index+999).execute().data)])
else:
    df = pd.DataFrame(supabase.table(table_name).select('*').execute().data)

df['actual_hit'] = (df['outcome_price'] > df['price_level']).astype(int)

results = []
for interval, group in df.groupby("interval"):
    group = group.dropna(subset=["prediction", "actual_hit"])
    if len(group) < 50 or group["actual_hit"].nunique() < 2:
        continue
    results.append({
        "interval": interval,
        "n": len(group),
        "hit_rate": group["actual_hit"].mean(),
        "mean_prediction": group["prediction"].mean(),
        "auc": roc_auc_score(group["actual_hit"], group["prediction"]),
        "avg_precision": average_precision_score(group["actual_hit"], group["prediction"])
    })

results_df = pd.DataFrame(results).sort_values("auc", ascending=False)
print(results_df)