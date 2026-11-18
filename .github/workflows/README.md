# .github/workflows

The scheduled and push-triggered jobs that keep LevelEdge running. Scheduled jobs run only from `main`, and every workflow can also be started by hand from the Actions tab.

| Workflow | Trigger (UTC) | Runs | Writes to |
|---|---|---|---|
| `run_forwardtest.yml` | 14:00, 16:00, 18:00 Mon–Fri | `scripts/forwardtest.py`, then `scripts/ledger_push.sh ... append --anchor` | Supabase `forwardtest_logs`; the `ledger` branch |
| `update_outcomes.yml` | Hourly | `scripts/update_outcomes.py` | Supabase `forwardtest_logs` |
| `paper_trade.yml` | 21:30 Mon–Fri | `scripts/paper_trade.py --execute`, `scripts/weekly_report.py` | The Alpaca paper account; the report issue |
| `forward_log.yml` | 22:00 Mon–Fri | `scripts/log_forward.py` | The `forward-log` branch |
| `ledger_reconcile.yml` | 22:30 Mon–Fri, 12:00 daily | `scripts/ledger_push.sh ... reconcile --anchor`, then `ledger.py verify --check-tsa` | The `ledger` branch |
| `sync_to_hf.yml` | Push to `main` | `scripts/sync_space.sh` | The HuggingFace Space |

Failures post an alert, with the reason, to the "Paper book weekly report" issue.

For secrets, alerts and what to do when a job fails, see [docs/operations.md](../../docs/operations.md).
