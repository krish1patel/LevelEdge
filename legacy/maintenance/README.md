# legacy/maintenance

One-off data maintenance scripts, kept for the record.

| Script | Purpose |
|---|---|
| `fix_intl_mins.py` | Backfilled the `interval_minutes` column of Supabase forward-test rows from each row's interval string. It writes to Supabase, so only run it deliberately. |
