import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Tuple


@dataclass
class PredictionRecord:
    raw: Dict[str, Any]
    logged_at: datetime
    prediction: float


@dataclass
class PredictionCluster:
    """Cluster of predictions that are close in time for the same key."""

    records: List[PredictionRecord] = field(default_factory=list)

    @property
    def start_time(self) -> datetime:
        return self.records[0].logged_at

    @property
    def end_time(self) -> datetime:
        return self.records[-1].logged_at

    def can_add(self, record: PredictionRecord, window: timedelta) -> bool:
        """
        Decide whether a new record belongs in this cluster.

        We consider predictions within `window` of the last record in the
        cluster to be "about the same time".
        """
        return abs((record.logged_at - self.end_time).total_seconds()) <= window.total_seconds()

    def add(self, record: PredictionRecord) -> None:
        self.records.append(record)

    def to_summary(self) -> Dict[str, Any]:
        """Summarize all records in this cluster into a single JSON-serializable dict."""
        base = self.records[0].raw.copy()

        # Basic stats on predictions
        preds = [r.prediction for r in self.records]
        avg_pred = sum(preds) / len(preds)

        # Aggregate numeric fields that are likely to be useful
        def safe_avg(key: str) -> float:
            vals = [float(r.raw[key]) for r in self.records if key in r.raw]
            return sum(vals) / len(vals) if vals else None

        # Collect simple aggregates
        summary: Dict[str, Any] = {
            "ticker": base.get("ticker"),
            "is_crypto": base.get("is_crypto"),
            "price_level": base.get("price_level"),
            "target_datetime": base.get("target_datetime"),
            # Time window of this cluster
            "logged_at_utc_start": self.start_time.isoformat(),
            "logged_at_utc_end": self.end_time.isoformat(),
            # Prediction statistics
            "prediction_avg": avg_pred,
            "prediction_min": min(preds),
            "prediction_max": max(preds),
            "prediction_count": len(preds),
            # Averages of other numeric fields
            "current_price_avg": safe_avg("current_price"),
            "target_price_ratio_avg": safe_avg("target_price_ratio"),
            "candles_ahead_avg": safe_avg("candles_ahead"),
            # Interval info across underlying predictions
            "intervals": sorted({r.raw.get("interval") for r in self.records if "interval" in r.raw}),
            "interval_minutes_values": sorted(
                {r.raw.get("interval_minutes") for r in self.records if "interval_minutes" in r.raw}
            ),
            # Underlying predictions for reference
            "original_predictions": [
                {
                    "logged_at_utc": r.raw.get("logged_at_utc"),
                    "interval": r.raw.get("interval"),
                    "interval_minutes": r.raw.get("interval_minutes"),
                    "prediction": r.raw.get("prediction"),
                    "candles_ahead": r.raw.get("candles_ahead"),
                }
                for r in self.records
            ],
        }

        # Remove None values for cleanliness
        return {k: v for k, v in summary.items() if v is not None}


def parse_timestamp(ts: str) -> datetime:
    """Parse ISO-8601 timestamp string with timezone into datetime."""
    # Python 3.11's fromisoformat handles the format in prediction_logs.jsonl
    return datetime.fromisoformat(ts)


def iter_records(f) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file handle, skipping blank lines."""
    for line in f:
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def build_clusters(
    records: Iterable[Dict[str, Any]],
    time_window: timedelta,
) -> Iterable[Tuple[Tuple[Any, ...], PredictionCluster]]:
    """
    Group records into clusters of "about the same time".

    Key for grouping is (ticker, price_level, target_datetime) so we don't
    accidentally merge different target times.
    """
    clusters_by_key: Dict[Tuple[Any, ...], List[PredictionCluster]] = defaultdict(list)

    for rec in records:
        logged_at_str = rec.get("logged_at_utc")
        if logged_at_str is None:
            # Skip malformed entries
            continue

        logged_at = parse_timestamp(logged_at_str)
        prediction_val = rec.get("prediction")
        if prediction_val is None:
            continue

        key = (
            rec.get("ticker"),
            rec.get("price_level"),
            rec.get("target_datetime"),
        )

        pr = PredictionRecord(raw=rec, logged_at=logged_at, prediction=float(prediction_val))

        key_clusters = clusters_by_key[key]
        if not key_clusters:
            key_clusters.append(PredictionCluster(records=[pr]))
        else:
            last_cluster = key_clusters[-1]
            if last_cluster.can_add(pr, time_window):
                last_cluster.add(pr)
            else:
                key_clusters.append(PredictionCluster(records=[pr]))

    # Flatten out as (key, cluster) pairs
    for key, clist in clusters_by_key.items():
        for cluster in clist:
            yield key, cluster


def postprocess_jsonl(
    input_path: str,
    output_path: str,
    time_window_seconds: float,
) -> None:
    """
    Read predictions from `input_path`, cluster by (ticker, price_level, target_datetime)
    and timestamp proximity, then write summarized clusters to `output_path` as CSV.
    """
    time_window = timedelta(seconds=time_window_seconds)

    with open(input_path, "r", encoding="utf-8") as f_in:
        clusters = list(build_clusters(iter_records(f_in), time_window))

    # Sort clusters chronologically by their start time for stable output
    clusters.sort(key=lambda item: item[1].start_time)

    # Prepare rows for CSV
    summaries = [cluster.to_summary() for _, cluster in clusters]

    if not summaries:
        # Nothing to write; still create an empty file with header?
        with open(output_path, "w", newline="", encoding="utf-8") as f_out:
            pass
        return

    # Define a stable set/order of CSV columns
    fieldnames = [
        "ticker",
        "is_crypto",
        "price_level",
        "target_datetime",
        "logged_at_utc_start",
        "logged_at_utc_end",
        "prediction_avg",
        "prediction_min",
        "prediction_max",
        "prediction_count",
        "current_price_avg",
        "target_price_ratio_avg",
        "candles_ahead_avg",
        "intervals",
        "interval_minutes_values",
        "original_predictions",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(fieldnames=fieldnames, f=f_out)
        writer.writeheader()
        for summary in summaries:
            row = summary.copy()
            # Serialize list-like fields as JSON strings so they fit in a CSV cell
            for key in ("intervals", "interval_minutes_values", "original_predictions"):
                if key in row:
                    row[key] = json.dumps(row[key])
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process prediction JSONL logs by clustering predictions that "
            "occur at about the same time for the same ticker and price level, "
            "then averaging their prediction values."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        default="prediction_logs.jsonl",
        help="Path to input JSONL file (default: prediction_logs.jsonl)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="prediction_logs_postprocessed.csv",
        help="Path to output CSV file (default: prediction_logs_postprocessed.csv)",
    )
    parser.add_argument(
        "--time-window-seconds",
        "-w",
        type=float,
        default=5.0,
        help=(
            "Maximum time difference (in seconds) between predictions to treat them "
            "as occurring 'about the same time' (default: 5.0)"
        ),
    )

    args = parser.parse_args()
    postprocess_jsonl(
        input_path=args.input,
        output_path=args.output,
        time_window_seconds=args.time_window_seconds,
    )


if __name__ == "__main__":
    main()

