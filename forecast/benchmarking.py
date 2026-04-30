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
]

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

DIRECT_MICRO_FOLDS = [("2015-W01", "2021-W52", "2022-W01", "2022-W13")]
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


def benchmark_variants(no_artifacts: bool = False, suite: str = "all") -> pd.DataFrame:
    cfg = Config()
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    variants = {
        "all": DEFAULT_VARIANTS,
        "lv2": LV2_VARIANTS,
        "direct": DIRECT_VARIANTS,
        "direct_micro": DIRECT_MICRO_VARIANTS,
    }.get(suite, DEFAULT_VARIANTS)
    folds = DIRECT_MICRO_FOLDS if suite == "direct_micro" else None

    print("benchmark: build marts", flush=True)
    sample = validate_inputs(cfg)
    daily = build_daily_mart(cfg, sample)
    weekly = build_weekly_mart(daily)

    rows = []
    for name, env in variants:
        print(f"benchmark: variant start {name} env={env}", flush=True)
        with variant_environment(env):
            metrics = run_backtests(daily, weekly, include_event_floor_table=False, folds=folds)
        metrics = metrics.copy()
        metrics.insert(0, "variant", name)
        for key in VARIANT_ENV_KEYS:
            metrics[key] = env.get(key, "")
        rows.append(metrics)

        out = pd.concat(rows, ignore_index=True)
        if not no_artifacts:
            out.to_csv(cfg.artifact_dir / "separate_benchmark_metrics.csv", index=False)
        print(benchmark_summary(out).to_string(index=False), flush=True)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def benchmark_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "variant",
        "fold",
        "weekly_revenue_wape",
        "weekly_cogs_wape",
        "daily_revenue_wape",
        "daily_cogs_wape",
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
        choices=["all", "lv2", "direct", "direct_micro"],
        default="all",
        help="Variant suite to run",
    )
    args = parser.parse_args()
    result = benchmark_variants(no_artifacts=args.no_artifacts, suite=args.suite)
    print("\nbenchmark: final summary")
    print(benchmark_summary(result).to_string(index=False))


if __name__ == "__main__":
    main()
