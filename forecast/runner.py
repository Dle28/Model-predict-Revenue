from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from .backtesting import run_backtests, run_yearly_rolling_origin_cv
from .common import Config, add_calendar_columns, checkpoint, validate_inputs
from .diagnostics import recovery_anchor_diagnostic, recovery_anchor_summary, yearly_revenue_diagnostics
from .final import coherence_summary, final_forecast, validate_submission
from .marts import build_daily_mart, build_weekly_mart
from .plotting import save_recovery_anchor_plot, save_submission_plot


BEST650_BASE_SUBMISSION = "submission_best_650k.csv"
BEST650_VARIANT_SUBMISSION = "submission_best650_rev1000_cogs1015.csv"
RAW_MODEL_SUBMISSION = "submission_model_raw.csv"
BEST650_REVENUE_SCALE = 1.0
BEST650_COGS_SCALE = 1.015

FORECAST_ENV_KEYS = [
    "FORECAST_LV1_DIRECT_ANCHOR_WEIGHT",
    "FORECAST_LV1_DIRECT_ANCHOR_HORIZONS",
    "FORECAST_LV2_HIST_DECAY_HALFLIFE_WEEKS",
    "FORECAST_DIRECT_BLEND_WEIGHT",
    "FORECAST_WEEKLY_DRIFT_CAP",
    "FORECAST_LV3_EVENT_BASE_MODE",
    "FORECAST_LV3_EVENT_FLOOR",
    "FORECAST_STRUCTURAL_WEEKLY_WEIGHT",
    "FORECAST_MODEL_BACKEND",
    "FORECAST_LV3_MODEL_BACKEND",
    "FORECAST_PROMO_KNOWN_END_DATE",
    "FORECAST_PROMO_PROJECTION",
    "FORECAST_LV1_USE_PROJECTED_PROMO",
    "FORECAST_CAPACITY_TREND_FLOOR",
    "FORECAST_RECOVERY_WEIGHT_BASE",
    "FORECAST_RECOVERY_WEIGHT_SLOPE",
    "FORECAST_RECOVERY_MODEL_WEIGHT_BASE",
    "FORECAST_RECOVERY_MODEL_WEIGHT_SLOPE",
    "FORECAST_RECOVERY_WEIGHT_NORMALIZATION_BASE",
    "FORECAST_RECOVERY_WEIGHT_NORMALIZATION_SLOPE",
    "FORECAST_MODEL_WEIGHT_NORMALIZATION_BASE",
    "FORECAST_MODEL_WEIGHT_NORMALIZATION_SLOPE",
    "FORECAST_CONTROLLED_RECOVERY_REVENUE_TOTAL",
]

EXTRA_OUTPUT_PATTERNS = [
    "best650_*_variant_summary.csv",
    "recovery_*_variant_summary.csv",
    "recovery_variant_summary.csv",
    "run_recovery_*.log",
    "submission_best650_*.csv",
    "submission_recovery_*.csv",
    "submission_plot_recovery_*.html",
]

EXTRA_ARTIFACT_PATTERNS = [
    "backtest_metrics_candidate*.csv",
    "*.stdout.log",
    "*.stderr.log",
    "separate_benchmark_metrics*.csv",
    "model_validation_summary.md",
    "yearly_rolling_origin_cv.csv",
]


def save_artifact(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"saved {path} ({len(df)} rows)")


def use_canonical_environment() -> None:
    for key in FORECAST_ENV_KEYS:
        os.environ.pop(key, None)
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["FORECAST_MODEL_BACKEND"] = "xgboost"
    os.environ["FORECAST_LV3_MODEL_BACKEND"] = "xgboost"


def clean_generated_files(cfg: Config, clean_outputs: bool, clean_artifacts: bool) -> list[Path]:
    deleted: list[Path] = []
    keep_outputs = {
        cfg.submission_name,
        RAW_MODEL_SUBMISSION,
        BEST650_BASE_SUBMISSION,
        BEST650_VARIANT_SUBMISSION,
        "submission_intervals.csv",
        "submission_plot.html",
        "recovery_anchor_plot.html",
    }
    keep_artifacts = {
        "daily_mart.csv",
        "weekly_mart.csv",
        "weekly_forecast.csv",
        "recovery_anchor_diagnostic.csv",
        "yearly_revenue_diagnostic.csv",
        "backtest_metrics.csv",
        "event_floor_validation.csv",
        "pipeline_best650_summary.md",
    }
    if clean_outputs:
        for pattern in EXTRA_OUTPUT_PATTERNS:
            for path in cfg.output_dir.glob(pattern):
                if path.is_file() and path.name not in keep_outputs:
                    path.unlink()
                    deleted.append(path)
    if clean_artifacts:
        for pattern in EXTRA_ARTIFACT_PATTERNS:
            for path in cfg.artifact_dir.glob(pattern):
                if path.is_file() and path.name not in keep_artifacts:
                    path.unlink()
                    deleted.append(path)
    return deleted


def calibrated_best650_submission(cfg: Config) -> pd.DataFrame:
    base_path = cfg.output_dir / BEST650_BASE_SUBMISSION
    expected = ["Date", "Revenue", "COGS"]
    if base_path.exists():
        submission = pd.read_csv(base_path)
        if list(submission.columns) != expected:
            raise ValueError(f"{base_path} columns must be {expected}, got {list(submission.columns)}")
        submission["Revenue"] = (pd.to_numeric(submission["Revenue"], errors="raise") * BEST650_REVENUE_SCALE).round(2)
        submission["COGS"] = (pd.to_numeric(submission["COGS"], errors="raise") * BEST650_COGS_SCALE).round(2)
        return submission

    variant_path = cfg.output_dir / BEST650_VARIANT_SUBMISSION
    if variant_path.exists():
        submission = pd.read_csv(variant_path)
        if list(submission.columns) != expected:
            raise ValueError(f"{variant_path} columns must be {expected}, got {list(submission.columns)}")
        for col in ["Revenue", "COGS"]:
            submission[col] = pd.to_numeric(submission[col], errors="raise").round(2)
        return submission

    raise FileNotFoundError(
        f"Missing canonical best650 files: {base_path} and {variant_path}. "
        "Keep either the unscaled base or the final canonical variant to reproduce the best650 submission."
    )


def align_intervals_to_submission(intervals: pd.DataFrame, submission: pd.DataFrame) -> pd.DataFrame:
    out = intervals.copy()
    final = submission.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    final["Date"] = pd.to_datetime(final["Date"]).dt.strftime("%Y-%m-%d")
    out = out.merge(final, on="Date", how="left", suffixes=("", "_final"))
    for target in ["Revenue", "COGS"]:
        final_col = f"{target}_final"
        if final_col not in out.columns:
            continue
        original = pd.to_numeric(out[target], errors="coerce").replace([np.inf, -np.inf], np.nan)
        desired = pd.to_numeric(out[final_col], errors="coerce")
        ratio = (desired / original).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        for col in [target, f"{target}_p10", f"{target}_p90", f"{target}_p05", f"{target}_p95"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce") * ratio
        out[target] = desired
    out = out.drop(columns=[col for col in out.columns if col.endswith("_final")])
    for target in ["Revenue", "COGS"]:
        if {target, f"{target}_p10", f"{target}_p90", f"{target}_p05", f"{target}_p95"}.issubset(out.columns):
            out[f"{target}_p10"] = np.minimum(out[f"{target}_p10"], out[target])
            out[f"{target}_p90"] = np.maximum(out[f"{target}_p90"], out[target])
            out[f"{target}_p05"] = np.minimum(out[f"{target}_p05"], out[f"{target}_p10"])
            out[f"{target}_p95"] = np.maximum(out[f"{target}_p95"], out[f"{target}_p90"])
    for col in [col for col in out.columns if col != "Date"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").clip(lower=0).round(2)
    return out


def align_weekly_pred_to_submission(submission: pd.DataFrame, weekly_pred: pd.DataFrame) -> pd.DataFrame:
    if weekly_pred.empty:
        return weekly_pred
    daily = submission.copy()
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = add_calendar_columns(daily.rename(columns={"Date": "date"}), "date")
    weekly_sum = daily.groupby("week_id", as_index=False).agg(
        revenue_w_pred=("Revenue", "sum"),
        cogs_w_pred=("COGS", "sum"),
        sample_days=("date", "count"),
    )
    out = weekly_pred.drop(columns=["revenue_w_pred", "cogs_w_pred", "sample_days"], errors="ignore")
    return out.merge(weekly_sum, on="week_id", how="left")


def write_pipeline_summary(
    cfg: Config,
    raw_submission: pd.DataFrame,
    final_submission: pd.DataFrame,
    deleted_files: list[Path],
    ran_shap: bool,
) -> None:
    summary_path = cfg.artifact_dir / "pipeline_best650_summary.md"
    lines = [
        "# Best650 rev1000 cogs1015 pipeline summary",
        "",
        f"- Final submission: `{cfg.final_submission_path}`",
        f"- Canonical variant: `{cfg.output_dir / BEST650_VARIANT_SUBMISSION}`",
        f"- Raw model output: `{cfg.output_dir / RAW_MODEL_SUBMISSION}`",
        f"- Intervals: `{cfg.output_dir / 'submission_intervals.csv'}`",
        f"- SHAP directory: `{cfg.artifact_dir / 'shap'}`" if ran_shap else "- SHAP: skipped",
        "",
        "## Totals",
        "",
        "| file | rows | Revenue | COGS |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| raw model | {len(raw_submission)} | {raw_submission['Revenue'].sum():.2f} | "
            f"{raw_submission['COGS'].sum():.2f} |"
        ),
        (
            f"| final | {len(final_submission)} | {final_submission['Revenue'].sum():.2f} | "
            f"{final_submission['COGS'].sum():.2f} |"
        ),
        "",
        "## Calibration",
        "",
        f"- Source base: `{cfg.output_dir / BEST650_BASE_SUBMISSION}`",
        f"- Revenue scale: `{BEST650_REVENUE_SCALE:.3f}`",
        f"- COGS scale: `{BEST650_COGS_SCALE:.3f}`",
    ]
    if deleted_files:
        lines += ["", "## Cleaned Files", ""]
        lines += [f"- `{path}`" for path in deleted_files]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {summary_path}")


def run_shap_explanations(cfg: Config, shap_date: str, sample_size: int) -> None:
    output_dir = cfg.artifact_dir / "shap"
    output_dir.mkdir(parents=True, exist_ok=True)
    for target in ["revenue", "cogs"]:
        command = [
            sys.executable,
            "-m",
            "forecast.explain",
            "--target",
            target,
            "--date",
            shap_date,
            "--sample-size",
            str(sample_size),
            "--output-dir",
            str(output_dir),
        ]
        print("$ " + " ".join(command))
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Base + Spikes weekly-baseline to daily forecast pipeline")
    parser.add_argument("--no-artifacts", action="store_true", help="Run without saving intermediate mart artifacts")
    parser.add_argument(
        "--use-cached-marts",
        action="store_true",
        help="Load artifacts/daily_mart.csv and artifacts/weekly_mart.csv instead of rebuilding marts",
    )
    parser.add_argument("--skip-backtest", action="store_true", help="Skip rolling-origin backtest and event-floor validation")
    parser.add_argument("--with-backtest", dest="skip_backtest", action="store_false", help="Run rolling-origin backtest")
    parser.add_argument("--yearly-cv", action="store_true", help="Run yearly rolling-origin CV folds for reporting")
    parser.add_argument("--raw-model-only", action="store_true", help="Save raw model output instead of best650 calibrated output")
    parser.add_argument("--respect-env", action="store_true", help="Do not reset forecast env vars to canonical best650 defaults")
    parser.add_argument("--skip-shap", action="store_true", help="Skip LV3 SHAP artifact generation")
    parser.add_argument("--shap-date", default="2021-11-11", help="Training date for SHAP waterfall plots")
    parser.add_argument("--shap-sample-size", type=int, default=800, help="Rows for SHAP summary plots")
    parser.add_argument("--clean-extra-outputs", action="store_true", help="Remove redundant generated output variants")
    parser.add_argument("--clean-extra-artifacts", action="store_true", help="Remove redundant generated benchmark logs")
    args = parser.parse_args()

    cfg = Config()
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    deleted_files = clean_generated_files(cfg, args.clean_extra_outputs, args.clean_extra_artifacts)
    if deleted_files:
        print(f"deleted {len(deleted_files)} redundant generated files")

    if not args.respect_env:
        use_canonical_environment()

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
    raw_submission, weekly_pred, intervals = final_forecast(daily, weekly, sample)
    submission = raw_submission.copy()
    if not args.raw_model_only:
        checkpoint("Checkpoint 6: apply best650 rev1000 cogs1015 calibration")
        submission = calibrated_best650_submission(cfg)
        intervals = align_intervals_to_submission(intervals, submission)
        weekly_pred = align_weekly_pred_to_submission(submission, weekly_pred)

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
        save_artifact(raw_submission, cfg.output_dir / RAW_MODEL_SUBMISSION)
    save_artifact(submission, cfg.final_submission_path)
    if not args.raw_model_only:
        save_artifact(submission, cfg.output_dir / BEST650_VARIANT_SUBMISSION)
    save_recovery_anchor_plot(recovery_diag, cfg.output_dir / "recovery_anchor_plot.html")
    save_submission_plot(
        submission,
        intervals,
        daily[daily["revenue"].notna()][["date", "revenue", "cogs"]],
        metrics,
        cfg.submission_plot_path,
    )
    if not args.skip_shap:
        checkpoint("Checkpoint 7: SHAP explanations")
        run_shap_explanations(cfg, args.shap_date, args.shap_sample_size)
    if not args.no_artifacts:
        write_pipeline_summary(cfg, raw_submission, submission, deleted_files, ran_shap=not args.skip_shap)
    print("done")


if __name__ == "__main__":
    main()
