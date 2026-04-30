from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtesting import run_backtests, run_yearly_rolling_origin_cv
from .common import Config, checkpoint, validate_inputs
from .diagnostics import recovery_anchor_diagnostic, recovery_anchor_summary, yearly_revenue_diagnostics
from .final import coherence_summary, final_forecast, validate_submission
from .marts import build_daily_mart, build_weekly_mart
from .plotting import save_recovery_anchor_plot, save_submission_plot


def save_artifact(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"saved {path} ({len(df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Base + Spikes weekly-baseline to daily forecast pipeline")
    parser.add_argument("--no-artifacts", action="store_true", help="Run without saving intermediate mart artifacts")
    parser.add_argument(
        "--use-cached-marts",
        action="store_true",
        help="Load artifacts/daily_mart.csv and artifacts/weekly_mart.csv instead of rebuilding marts",
    )
    parser.add_argument("--skip-backtest", action="store_true", help="Skip rolling-origin backtest and event-floor validation")
    parser.add_argument("--yearly-cv", action="store_true", help="Run yearly rolling-origin CV folds for reporting")
    args = parser.parse_args()

    cfg = Config()
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint("Checkpoint 0: validate inputs")
    sample = validate_inputs(cfg)

    if args.use_cached_marts:
        checkpoint("Checkpoint 1-2: load cached marts")
        daily_path = cfg.artifact_dir / "daily_mart.csv"
        weekly_path = cfg.artifact_dir / "weekly_mart.csv"
        if not daily_path.exists() or not weekly_path.exists():
            missing = [str(path) for path in [daily_path, weekly_path] if not path.exists()]
            raise FileNotFoundError(f"Missing cached mart file(s): {missing}")
        daily = pd.read_csv(daily_path, parse_dates=["date", "week_start"], low_memory=False)
        weekly = pd.read_csv(weekly_path, parse_dates=["week_start"], low_memory=False)
        print("daily_mart:", len(daily), "rows", daily["date"].min().date(), "->", daily["date"].max().date())
        print("weekly_mart:", len(weekly), "rows", weekly["week_id"].min(), "->", weekly["week_id"].max())
    else:
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

    checkpoint("Checkpoint 2b: yearly revenue regime diagnostic")
    yearly_diag = yearly_revenue_diagnostics(daily)
    print(yearly_diag.to_string(index=False))
    if not args.no_artifacts:
        save_artifact(yearly_diag, cfg.artifact_dir / "yearly_revenue_diagnostic.csv")

    if args.skip_backtest:
        checkpoint("Checkpoint 3: skip rolling-origin backtest")
        metrics = pd.DataFrame()
        event_floor_validation = pd.DataFrame()
    else:
        checkpoint("Checkpoint 3: rolling-origin backtest")
        metrics, event_floor_validation = run_backtests(daily, weekly, include_event_floor_table=True)
        print(metrics.to_string(index=False))
        if len(event_floor_validation):
            print("\nevent_floor_validation:")
            print(event_floor_validation.to_string(index=False))
        if not args.no_artifacts:
            save_artifact(metrics, cfg.artifact_dir / "backtest_metrics.csv")
            save_artifact(event_floor_validation, cfg.artifact_dir / "event_floor_validation.csv")

    if args.yearly_cv:
        checkpoint("Checkpoint 3b: yearly rolling-origin CV")
        yearly_cv = run_yearly_rolling_origin_cv(daily, weekly)
        print(yearly_cv.to_string(index=False))
        if not args.no_artifacts:
            save_artifact(yearly_cv, cfg.artifact_dir / "yearly_rolling_origin_cv.csv")

    checkpoint("Checkpoint 4-5: final Base + Spikes forecast and validation")
    submission, weekly_pred, intervals = final_forecast(daily, weekly, sample)

    if "revenue_lv3_backend" in weekly_pred.columns:
        backend = weekly_pred["revenue_lv3_backend"].dropna()
        if len(backend):
            print(f"LV3 Backend: {backend.iloc[0]}")
    if "revenue_w_lv1_backend" in weekly_pred.columns:
        backend = weekly_pred["revenue_w_lv1_backend"].dropna()
        if len(backend):
            print(f"LV1 Backend: {backend.iloc[0]}")

    validate_submission(submission, sample)
    coherence = coherence_summary(submission, weekly_pred)
    recovery_diag = recovery_anchor_diagnostic(weekly_pred)
    recovery_summary = recovery_anchor_summary(recovery_diag)
    if len(recovery_summary):
        print("\nrecovery_anchor_summary:")
        print(recovery_summary.to_string(index=False))
    if len(coherence):
        print(
            "bottom-up max drift:",
            "Revenue",
            f"{coherence['revenue_bottomup_drift'].max():.8f}",
            "COGS",
            f"{coherence['cogs_bottomup_drift'].max():.8f}",
            "| weekly drift p90/p95:",
            "Revenue",
            f"{coherence['revenue_weekly_drift'].quantile(0.90):.4f}",
            f"{coherence['revenue_weekly_drift'].quantile(0.95):.4f}",
            "COGS",
            f"{coherence['cogs_weekly_drift'].quantile(0.90):.4f}",
            f"{coherence['cogs_weekly_drift'].quantile(0.95):.4f}",
            "| max LV3 uplift:",
            "Revenue",
            f"{coherence['revenue_lv3_uplift'].max():.4f}",
            "COGS",
            f"{coherence['cogs_lv3_uplift'].max():.4f}",
        )
    if not args.no_artifacts:
        save_artifact(weekly_pred, cfg.artifact_dir / "weekly_forecast.csv")
        save_artifact(recovery_diag, cfg.artifact_dir / "recovery_anchor_diagnostic.csv")
        save_artifact(intervals, cfg.output_dir / "submission_intervals.csv")
    save_artifact(submission, cfg.final_submission_path)
    save_recovery_anchor_plot(recovery_diag, cfg.output_dir / "recovery_anchor_plot.html")
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
