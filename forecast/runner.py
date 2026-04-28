from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtesting import run_backtests
from .common import Config, checkpoint, validate_inputs
from .final import coherence_summary, final_forecast, validate_submission
from .marts import build_daily_mart, build_weekly_mart
from .plotting import save_submission_plot


def save_artifact(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"saved {path} ({len(df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-down weekly-to-daily forecast pipeline")
    parser.add_argument("--no-artifacts", action="store_true", help="Run without saving intermediate mart artifacts")
    args = parser.parse_args()

    cfg = Config()
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint("Checkpoint 0: validate inputs")
    sample = validate_inputs(cfg)

    checkpoint("Checkpoint 1: build daily mart")
    daily = build_daily_mart(cfg, sample)
    print("daily_mart:", len(daily), "rows", daily["date"].min().date(), "->", daily["date"].max().date())
    if not args.no_artifacts:
        save_artifact(daily, cfg.artifact_dir / "daily_mart.csv")

    checkpoint("Checkpoint 2: build weekly mart")
    weekly = build_weekly_mart(daily)
    print("weekly_mart:", len(weekly), "rows", weekly["week_id"].min(), "->", weekly["week_id"].max())
    if not args.no_artifacts:
        save_artifact(weekly, cfg.artifact_dir / "weekly_mart.csv")

    checkpoint("Checkpoint 3: rolling-origin backtest")
    metrics = run_backtests(daily, weekly)
    print(metrics.to_string(index=False))
    if not args.no_artifacts:
        save_artifact(metrics, cfg.artifact_dir / "backtest_metrics.csv")

    checkpoint("Checkpoint 4-5: final forecast, reconciliation, validation")
    submission, weekly_pred, intervals = final_forecast(daily, weekly, sample)
    validate_submission(submission, sample)
    coherence = coherence_summary(submission, weekly_pred)
    if len(coherence):
        print(
            "coherence max drift:",
            "Revenue",
            f"{coherence['revenue_drift'].max():.8f}",
            "COGS",
            f"{coherence['cogs_drift'].max():.8f}",
        )
    if not args.no_artifacts:
        save_artifact(weekly_pred, cfg.artifact_dir / "weekly_forecast.csv")
        save_artifact(intervals, cfg.output_dir / "submission_intervals.csv")
    save_artifact(submission, cfg.final_submission_path)
    save_submission_plot(
        submission,
        intervals,
        daily[daily["revenue"].notna()][["date", "revenue", "cogs"]],
        metrics,
        cfg.submission_plot_path,
    )
    print("done")


if __name__ == "__main__":
    main()
