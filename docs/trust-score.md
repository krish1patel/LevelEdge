# Trust score

The research-integrity engine grades how far a strategy's backtest can be trusted, from 0 to 100, and lists the specific ways it might be fooling you. It runs every strategy in `leveledge.integrity.STRATEGIES` through eight checks. The Research page shows the results.

## Running the engine

### Locally

```bash
python scripts/integrity.py run --workers 8        # every strategy, about 30 s on 8 processes
python scripts/integrity.py run --strategies book  # one strategy
python scripts/integrity.py rescore                # re-score saved task results after a scoring change
```

### On AWS

```bash
python scripts/integrity.py submit                 # 135 tasks onto the leveledge-scan queue
python scripts/integrity.py collect --run-id RUN   # once all tasks are stored in DynamoDB
```

Both write `benchmarks/results/integrity/summary.json` (scores) and `tasks.json.gz` (raw task results). On AWS the tasks run on the market-scan workers: each result is stored once with a conditional write, and failed tasks are retried and then moved to the dead-letter queue. boto3 can't read an `aws login` session directly, so export it first: `eval "$(aws configure export-credentials --format env)"`.

## How a strategy is evaluated

The engine splits each strategy into independent tasks, so they run in any order on any machine:

| Task | Count per strategy | Computes |
|---|---|---|
| `base` | 1 | The backtest, its benchmark, a block-bootstrap Sharpe interval, 4× costs, the best-year test, and the look-ahead truncation test |
| `grid` | One per variant | Each variant's Sharpe before and after 2020 |
| `shuffle` | 4 (of 50 each) | Sharpe of 200 copies of the strategy's own holdings, shuffled in 4-week blocks |

## The eight checks

| Check | Points | Passes when |
|---|---|---|
| Still works on unseen years | 30 | Its 2020+ Sharpe is at least 60% of its pre-2020 Sharpe and of the benchmark's 2020+ Sharpe |
| Beats a simple alternative | 15 | The 95% interval of its Sharpe difference against the benchmark is above zero |
| Timing beats chance | 15 | It beats at least 95% of its time-shuffled copies |
| Survives counting every try | 10 | Its deflated Sharpe probability (Bailey & López de Prado 2014), counting every variant, is at least 95% |
| Variants keep their order | 15 | Variants' rank correlation before and after 2020 is at least 0.3 |
| Never sees the future | 5 | Decisions recomputed on data cut at four past dates match exactly. A failure caps the score at 20. |
| Survives higher costs | 5 | At four times the trading costs it keeps at least 80% of its Sharpe |
| Not one lucky year | 5 | Its advantage over the benchmark stays positive without its best year |

Grades: A 80+, B 65+, C 50+, D 35+, F below 35.

## Validating the engine

- **A peeking strategy fails.** `tests/test_integrity.py` builds a strategy that reads next week's price. The look-ahead check catches it, and the score is capped at 20.
- **The known overfit is caught.** On the 80-variant search the engine found a rank correlation of −0.70 between periods (data through Nov 17, 2026) without being told, against the original study's −0.73, and graded it D.
- **The two executors agree.** The local and AWS runs use fixed seeds. On the same data (prices through Nov 17, 2026) they produced identical scores for all seven strategies. Scores move when new prices arrive, so compare runs on the same data.

## Adding a strategy

Add an entry to `STRATEGIES` in `src/leveledge/integrity.py`: a `decide(ctx, **params)` function returning weekly target weights, its default `params`, a `grid` of variants, and a `benchmark` (`"spy"` or `"sixty_forty"`). Then run the engine.
