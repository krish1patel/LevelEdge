import os
from supabase import create_client
from dotenv import load_dotenv
from leveledge.constants import ALLOWED_INTERVALS

load_dotenv()

supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

# Map interval string to minutes
interval_to_minutes = {}
for i in ALLOWED_INTERVALS:
    if 'm' in i:
        interval_to_minutes[i] = int(i[:-1])
    elif 'h' in i:
        interval_to_minutes[i] = int(i[:-1]) * 60
    elif 'd' in i:
        interval_to_minutes[i] = int(i[:-1]) * 1440

# Fetch all rows where interval_minutes is null
rows = supabase.table("forwardtest_logs") \
    .select("id, interval") \
    .is_("interval_minutes", "null") \
    .execute().data

print(f"Found {len(rows)} rows to update")

for row in rows:
    minutes = interval_to_minutes.get(row["interval"])
    if minutes is None:
        print(f"  Unknown interval: {row['interval']} — skipping")
        continue
    supabase.table("forwardtest_logs") \
        .update({"interval_minutes": minutes}) \
        .eq("id", row["id"]) \
        .execute()
    print(f"  Updated row {row['id']}: {row['interval']} → {minutes}m")

print("Done.")