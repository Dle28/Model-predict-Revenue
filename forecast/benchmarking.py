from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from typing import Dict, Iterator, List, Tuple

import pandas as pd

from .backtesting import run_backtests
from .common import Config, validate_inputs
from .marts import build_daily_mart, build_weekly_mart


VARIANT_ENV_KEYS = [
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
]

MART_ENV_KEYS = {
    "FORECAST_PROMO_KNOWN_END_DATE",
    "FORECAST_PROMO_PROJECTION",
    "FORECAST_LV1_USE_PROJECTED_PROMO",
}

DEFAULT_VARIANTS: List[Tuple[str, Dict[str, str]]] = [
    ("baseline_current", {}),
    (
        "lv1_direct_anchor_020_h13",
        {
            "FORECAST_LV1_DIRECT_ANCHOR_WEIGHT": "0.20",
            "FORECAST_LV1_DIRECT_ANCHOR_HORIZONS": "13",
        },
    ),
    ("lv2_decay_52w", {"FORECAST_LV2_HIST_DECAY_HALFLIFE_WEEKS": "52"}),
    ("lv2_decay_26w", {"FORECAST_LV2_HIST_DECAY_HALFLIFE_WEEKS": "26"}),
]

LV2_VARIANTS: List[Tuple[str, Dict[str, str]]] = [
    ("baseline_current", {}),
    ("lv2_decay_52w", {"FORECAST_LV2_HIST_DECAY_HALFLIFE_WEEKS": "52"}),
    ("lv2_decay_26w", {"FORECAST_LV2_HIST_DECAY_HALFLIFE_WEEKS": "26"}),
]

DIRECT_VARIANTS: List[Tuple[str, Dict[str, str]]] = [
    ("direct_blend_000", {"FORECAST_DIRECT_BLEND_WEIGHT": "0.00"}),
    ("direct_blend_010", {"FORECAST_DIRECT_BLEND_WEIGHT": "0.10"}),
    ("direct_blend_015", {"FORECAST_DIRECT_BLEND_WEIGHT": "0.15"}),
    (
        "lv1_direct_anchor_020_h13",
        {
            "FORECAST_LV1_DIRECT_ANCHOR_WEIGHT": "0.20",
            "FORECAST_LV1_DIRECT_ANCHOR_HORIZONS": "13",
        },
    ),
]

DIRECT_MICRO_FOLDS = [("2019-W01", "2021-W52", "2022-W01", "2022-W13")]
EVENT_TUNING_FOLDS = [("2019-W01", "2021-W52", "2022-W40", "2022-W51")]
TUNING_2022_FOLDS = [("2019-W01", "2021-W52", "2022-W01", "2022-W51")]
DIRECT_MICRO_VARIANTS: List[Tuple[str, Dict[str, str]]] = [
    ("direct_blend_000_micro_13w", {"FORECAST_DIRECT_BLEND_WEIGHT": "0.00"}),
    ("direct_blend_010_micro_13w", {"FORECAST_DIRECT_BLEND_WEIGHT": "0.10"}),
    ("direct_blend_015_micro_13w", {"FORECAST_DIRECT_BLEND_WEIGHT": "0.15"}),
    (
        "lv1_direct_anchor_020_h13_micro_13w",
        {
            "FORECAST_LV1_DIRECT_ANCHOR_WEIGHT": "0.20",
            "FORECAST_LV1_DIRECT_ANCHOR_HORIZONS": "13",
        },
    ),
]

EVENT_TUNING_VARIANTS: List[Tuple[str, Dict[str, str]]] = [
    ("baseline_current", {}),
    ("direct_blend_005", {"FORECAST_DIRECT_BLEND_WEIGHT": "0.05"}),
    ("direct_blend_010", {"FORECAST_DIRECT_BLEND_WEIGHT": "0.10"}),
    ("drift_cap_005", {"FORECAST_WEEKLY_DRIFT_CAP": "0.05"}),
    ("drift_cap_015", {"FORECAST_WEEKLY_DRIFT_CAP": "0.15"}),
    ("lv2_decay_26w", {"FORECAST_LV2_HIST_DECAY_HALFLIFE_WEEKS": "26"}),
    ("lv2_decay_104w", {"FORECAST_LV2_HIST_DECAY_HALFLIFE_WEEKS": "104"}),
    ("event_weekly_avg", {"FORECAST_LV3_EVENT_BASE_MODE": "weekly_avg"}),
    (
        "event_weekly_avg_floor",
        {
            "FORECAST_LV3_EVENT_BASE_MODE": "weekly_avg",
            "FORECAST_LV3_EVENT_FLOOR": "on",
        },
    ),
]

TUNING_2022_VARIANTS: List[Tuple[str, Dict[str, str]]] = [
    ("baseline_current", {}),
    ("capacity_floor_on", {"FORECAST_CAPACITY_TREND_FLOOR": "1"}),
    ("direct_blend_005", {"FORECAST_DIRECT_BLEND_WEIGHT": "0.05"}),
    (
        "trend_blend_60",
        {
            "FORECAST_RECOVERY_WEIGHT_BASE": "0.10",
            "FORECAST_RECOVERY_WEIGHT_SLOPE": "0.08",
            "FORECAST_RECOVERY_MODEL_WEIGHT_BASE": "0.60",
            "FORECAST_RECOVERY_MODEL_WEIGHT_SLOPE": "0.05",
            "FORECAST_RECOVERY_WEIGHT_NORMALIZATION_BASE": "0.12",
            "FORECAST_RECOVERY_WEIGHT_NORMALIZATION_SLOPE": "0.08",
            "FORECAST_MODEL_WEIGHT_NORMALIZATION_BASE": "0.60",
            "FORECAST_MODEL_WEIGHT_NORMALIZATION_SLOPE": "0.05",
        },
    ),
    (
        "trend_blend_55",
        {
            "FORECAST_RECOVERY_WEIGHT_BASE": "0.12",
            "FORECAST_RECOVERY_WEIGHT_SLOPE": "0.10",
            "FORECAST_RECOVERY_MODEL_WEIGHT_BASE": "0.55",
            "FORECAST_RECOVERY_MODEL_WEIGHT_SLOPE": "0.05",
            "FORECAST_RECOVERY_WEIGHT_NORMALIZATION_BASE": "0.15",
            "FORECAST_RECOVERY_WEIGHT_NORMALIZATION_SLOPE": "0.10",
            "FORECAST_MODEL_WEIGHT_NORMALIZATION_BASE": "0.55",
            "FORECAST_MODEL_WEIGHT_NORMALIZATION_SLOPE": "0.05",
        },
    ),
    (
        "trend_blend_50",
        {
            "FORECAST_RECOVERY_WEIGHT_BASE": "0.15",
            "FORECAST_RECOVERY_WEIGHT_SLOPE": "0.10",
            "FORECAST_RECOVERY_MODEL_WEIGHT_BASE": "0.50",
            "FORECAST_RECOVERY_MODEL_WEIGHT_SLOPE": "0.05",
            "FORECAST_RECOVERY_WEIGHT_NORMALIZATION_BASE": "0.18",
            "FORECAST_RECOVERY_WEIGHT_NORMALIZATION_SLOPE": "0.10",
            "FORECAST_MODEL_WEIGHT_NORMALIZATION_BASE": "0.50",
            "FORECAST_MODEL_WEIGHT_NORMALIZATION_SLOPE": "0.05",
        },
    ),
    ("drift_cap_005", {"FORECAST_WEEKLY_DRIFT_CAP": "0.05"}),
    ("drift_cap_015", {"FORECAST_WEEKLY_DRIFT_CAP": "0.15"}),
    ("drift_cap_025", {"FORECAST_WEEKLY_DRIFT_CAP": "0.25"}),
    ("drift_cap_off", {"FORECAST_WEEKLY_DRIFT_CAP": "off"}),
    ("lv2_decay_26w", {"FORECAST_LV2_HIST_DECAY_HALFLIFE_WEEKS": "26"}),
]

DRIFT_CAP_VARIANTS: List[Tuple[str, Dict[str, str]]] = [
    ("drift_cap_default", {}),
    ("drift_cap_010", {"FORECAST_WEEKLY_DRIFT_CAP": "0.10"}),
    ("drift_cap_015", {"FORECAST_WEEKLY_DRIFT_CAP": "0.15"}),
    ("drift_cap_025", {"FORECAST_WEEKLY_DRIFT_CAP": "0.25"}),
    ("drift_cap_off", {"FORECAST_WEEKLY_DRIFT_CAP": "off"}),
]

PROMO_PROJECTION_VARIANTS: List[Tuple[str, Dict[str, str]]] = [
    ("actual_calendar", {}),
    (
        "asof_2021_no_projection",
        {
            "FORECAST_PROMO_KNOWN_END_DATE": "2021-12-31",
            "FORECAST_PROMO_PROJECTION": "0",
        },
    ),
    (
        "asof_2021_projection",
        {
            "FORECAST_PROMO_KNOWN_END_DATE": "2021-12-31",
            "FORECAST_PROMO_PROJECTION": "1",
        },
    ),
    (
        "asof_2021_projection_lv1",
        {
            "FORECAST_PROMO_KNOWN_END_DATE": "2021-12-31",
            "FORECAST_PROMO_PROJECTION": "1",
            "FORECAST_LV1_USE_PROJECTED_PROMO": "1",
        },
    ),
]

DIRECT_CONFIRM_VARIANTS: List[Tuple[str, Dict[str, str]]] = [
    ("direct_blend_000", {"FORECAST_DIRECT_BLEND_WEIGHT": "0.00"}),
    ("direct_blend_005", {"FORECAST_DIRECT_BLEND_WEIGHT": "0.05"}),
]

REDESIGN_2022_VARIANTS: List[Tuple[str, Dict[str, str]]] = [
    ("baseline_current", {"FORECAST_STRUCTURAL_WEEKLY_WEIGHT": "0.00"}),
    ("structural_w_010", {"FORECAST_STRUCTURAL_WEEKLY_WEIGHT": "0.10"}),
    ("structural_w_020", {"FORECAST_STRUCTURAL_WEEKLY_WEIGHT": "0.20"}),
    ("structural_w_035", {"FORECAST_STRUCTURAL_WEEKLY_WEIGHT": "0.35"}),
]

REDESIGN_CONFIRM_VARIANTS: List[Tuple[str, Dict[str, str]]] = [
    ("baseline_current", {"FORECAST_STRUCTURAL_WEEKLY_WEIGHT": "0.00"}),
    ("structural_w_010", {"FORECAST_STRUCTURAL_WEEKLY_WEIGHT": "0.10"}),
    ("structural_w_020", {"FORECAST_STRUCTURAL_WEEKLY_WEIGHT": "0.20"}),
]


@contextmanager
def variant_environment(values: Dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in VARIANT_ENV_KEYS}
    try:
        for key in VARIANT_ENV_KEYS:
            os.environ.pop(key, None)
        for key, value in values.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_or_build_marts(cfg: Config, use_cached_marts: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sample = validate_inputs(cfg)
    if use_cached_marts:
        daily_path = cfg.artifact_dir / "daily_mart.csv"
        weekly_path = cfg.artifact_dir / "weekly_mart.csv"
        if daily_path.exists() and weekly_path.exists():
            daily = pd.read_csv(daily_path, parse_dates=["date", "week_start"], low_memory=False)
            weekly = pd.read_csv(weekly_path, parse_dates=["week_start"], low_memory=False)
            return daily, weekly
    daily = build_daily_mart(cfg, sample)
    weekly = build_weekly_mart(daily)
    return daily, weekly


def benchmark_variants(no_artifacts: bool = False, suite: str = "all", use_cached_marts: bool = False) -> pd.DataFrame:
    cfg = Config()
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    variants = {
        "all": DEFAULT_VARIANTS,
        "lv2": LV2_VARIANTS,
        "direct": DIRECT_VARIANTS,
        "direct_micro": DIRECT_MICRO_VARIANTS,
        "event_tuning": EVENT_TUNING_VARIANTS,
        "tuning_2022": TUNING_2022_VARIANTS,
        "direct_confirm": DIRECT_CONFIRM_VARIANTS,
        "redesign_2022": REDESIGN_2022_VARIANTS,
        "redesign_confirm": REDESIGN_CONFIRM_VARIANTS,
        "drift_cap": DRIFT_CAP_VARIANTS,
        "promo_projection": PROMO_PROJECTION_VARIANTS,
    }.get(suite, DEFAULT_VARIANTS)
    folds = None
    if suite == "direct_micro":
        folds = DIRECT_MICRO_FOLDS
    elif suite == "event_tuning":
        folds = EVENT_TUNING_FOLDS
    elif suite in {"tuning_2022", "redesign_2022", "drift_cap", "promo_projection"}:
        folds = TUNING_2022_FOLDS

    print("benchmark: load cached marts" if use_cached_marts else "benchmark: build marts", flush=True)
    daily, weekly = _load_or_build_marts(cfg, use_cached_marts)

    rows = []
    for name, env in variants:
        print(f"benchmark: variant start {name} env={env}", flush=True)
        with variant_environment(env):
            if MART_ENV_KEYS.intersection(env):
                variant_daily, variant_weekly = _load_or_build_marts(cfg, use_cached_marts=False)
            else:
                variant_daily, variant_weekly = daily, weekly
            metrics = run_backtests(variant_daily, variant_weekly, include_event_floor_table=False, folds=folds)
        metrics = metrics.copy()
        if {"daily_revenue_wape", "daily_cogs_wape"}.issubset(metrics.columns):
            metrics["daily_wape_mean"] = (metrics["daily_revenue_wape"] + metrics["daily_cogs_wape"]) / 2.0
        if {"weekly_revenue_wape", "weekly_cogs_wape"}.issubset(metrics.columns):
            metrics["weekly_wape_mean"] = (metrics["weekly_revenue_wape"] + metrics["weekly_cogs_wape"]) / 2.0
        if {"daily_wape_mean", "weekly_wape_mean"}.issubset(metrics.columns):
            metrics["selection_score"] = 0.70 * metrics["daily_wape_mean"] + 0.30 * metrics["weekly_wape_mean"]
        metrics.insert(0, "variant", name)
        for key in VARIANT_ENV_KEYS:
            metrics[key] = env.get(key, "")
        rows.append(metrics)

        out = pd.concat(rows, ignore_index=True)
        if not no_artifacts:
            out.to_csv(cfg.artifact_dir / "separate_benchmark_metrics.csv", index=False)
            out.to_csv(cfg.artifact_dir / f"separate_benchmark_metrics_{suite}.csv", index=False)
        print(benchmark_summary(out).to_string(index=False), flush=True)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def benchmark_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "variant",
        "fold",
        "weekly_revenue_wape",
        "weekly_cogs_wape",
        "weekly_wape_mean",
        "daily_revenue_wape",
        "daily_cogs_wape",
        "daily_wape_mean",
        "selection_score",
        "daily_revenue_bias_all",
        "daily_cogs_bias_all",
        "daily_revenue_wape_high_bucket",
        "daily_revenue_bias_high_bucket",
        "daily_cogs_bias_high_bucket",
        "daily_revenue_p10_p90_coverage",
        "daily_cogs_p10_p90_coverage",
    ]
    return metrics[[col for col in cols if col in metrics.columns]].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated benchmark variants on the standard backtest fold")
    parser.add_argument("--no-artifacts", action="store_true", help="Do not save benchmark CSV artifacts")
    parser.add_argument(
        "--suite",
        choices=[
            "all",
            "lv2",
            "direct",
            "direct_micro",
            "event_tuning",
            "tuning_2022",
            "direct_confirm",
            "redesign_2022",
            "redesign_confirm",
            "drift_cap",
            "promo_projection",
        ],
        default="all",
        help="Variant suite to run",
    )
    parser.add_argument(
        "--use-cached-marts",
        action="store_true",
        help="Load artifacts/daily_mart.csv and artifacts/weekly_mart.csv instead of rebuilding marts",
    )
    args = parser.parse_args()
    result = benchmark_variants(no_artifacts=args.no_artifacts, suite=args.suite, use_cached_marts=args.use_cached_marts)
    print("\nbenchmark: final summary")
    print(benchmark_summary(result).to_string(index=False))


if __name__ == "__main__":
    main()
