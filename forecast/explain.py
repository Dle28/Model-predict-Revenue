from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .common import Config, FINAL_TRAIN_END_WEEK_ID, FINAL_TRAIN_START_WEEK_ID, validate_inputs
from .lv1 import historical_weekly_base
from .lv2 import allocate_base_daily_dynamic, fit_allocation_model, lv2_columns
from .lv3 import _prepare_base_join, fit_spike_multiplier_model, spike_features
from .marts import build_daily_mart, build_weekly_mart


def _load_shap():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap  # type: ignore

        return shap, plt
    except Exception as exc:
        raise RuntimeError(
            "SHAP is not installed in this environment. Install it with `python -m pip install shap` "
            "then rerun `python -m forecast.explain`."
        ) from exc


def _target_config(target: str) -> Tuple[str, str, str, str]:
    target = target.lower().strip()
    if target == "revenue":
        return "revenue", "revenue_w", "revenue_w_base", "revenue_lv2_base"
    if target == "cogs":
        return "cogs", "cogs_w", "cogs_w_base", "cogs_lv2_base"
    raise ValueError("target must be `revenue` or `cogs`")


def _fit_lv3_explain_frame(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    target: str,
    train_start_week_id: str,
    train_end_week_id: str,
) -> Tuple[object, pd.DataFrame, pd.DataFrame]:
    target_col, weekly_target_col, weekly_base_col, daily_base_col = _target_config(target)
    train_start_date = pd.Timestamp(weekly.loc[weekly["week_id"].eq(train_start_week_id), "week_start"].min())
    train_end_week_start = pd.Timestamp(weekly.loc[weekly["week_id"].eq(train_end_week_id), "week_start"].max())
    train_end_date = train_end_week_start + pd.Timedelta(days=6)

    hist_weekly_base = historical_weekly_base(weekly, weekly_target_col, train_end_week_id, train_start_week_id)
    allocation_model = fit_allocation_model(
        daily,
        target_col,
        hist_weekly_base,
        weekly_base_col,
        train_end_date,
        train_start_date,
    )
    train_daily = daily[
        daily["date"].between(train_start_date, train_end_date)
        & daily[target_col].notna()
        & daily["week_id"].isin(hist_weekly_base["week_id"])
    ].copy()
    train_base = allocate_base_daily_dynamic(
        train_daily[lv2_columns(train_daily)],
        hist_weekly_base,
        weekly_base_col,
        daily_base_col,
        allocation_model,
    )
    spike_model = fit_spike_multiplier_model(
        daily,
        target_col,
        train_base,
        daily_base_col,
        train_end_date,
        train_start_date,
    )
    joined = _prepare_base_join(train_daily, train_base, daily_base_col)
    joined = joined[joined["lv2_base_value"].notna() & joined["lv2_base_value"].gt(1e-9)].copy()
    features = spike_features(joined)
    return spike_model, joined, features


def _save_feature_contribution_table(explanation, features: pd.DataFrame, row_index: int, path: Path) -> None:
    values = np.asarray(explanation.values[row_index], dtype=float)
    out = pd.DataFrame(
        {
            "feature": features.columns,
            "feature_value": features.iloc[row_index].to_numpy(dtype=float),
            "shap_value_log_multiplier": values,
            "abs_shap_value": np.abs(values),
        }
    ).sort_values("abs_shap_value", ascending=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def _save_feature_importance(model: object, features: pd.DataFrame, path: Path) -> None:
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(getattr(model, "feature_importances_"), dtype=float)
    elif hasattr(model, "coef"):
        coef = getattr(model, "coef")
        importance = np.abs(np.asarray(coef, dtype=float).ravel()[-len(features.columns) :])
    else:
        importance = np.zeros(len(features.columns), dtype=float)
    out = pd.DataFrame(
        {
            "feature": features.columns,
            "importance": importance,
        }
    ).sort_values("importance", ascending=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAP explanations for the LV3 tree spike model")
    parser.add_argument("--target", choices=["revenue", "cogs"], default="revenue")
    parser.add_argument("--date", default="2021-11-11", help="Waterfall explanation date")
    parser.add_argument("--sample-size", type=int, default=800, help="Rows for SHAP summary plot")
    parser.add_argument("--output-dir", default="artifacts/shap", help="Directory for SHAP artifacts")
    args = parser.parse_args()

    cfg = Config()
    sample = validate_inputs(cfg)
    daily = build_daily_mart(cfg, sample)
    weekly = build_weekly_mart(daily)
    spike_model, joined, features = _fit_lv3_explain_frame(
        daily,
        weekly,
        args.target,
        FINAL_TRAIN_START_WEEK_ID,
        FINAL_TRAIN_END_WEEK_ID,
    )
    if getattr(spike_model, "backend", "") not in {"lightgbm_monotone", "xgboost_monotone"}:
        raise RuntimeError(f"SHAP TreeExplainer needs a tree LV3 backend, got {spike_model.backend!r}.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = spike_model.model
    importance_path = output_dir / f"{args.target}_lv3_feature_importance.csv"
    _save_feature_importance(model, features, importance_path)

    try:
        shap, plt = _load_shap()
    except RuntimeError as exc:
        print(f"saved {importance_path}")
        print(str(exc))
        return

    summary_X = features.tail(max(args.sample_size, 1))
    explainer = shap.TreeExplainer(model)
    summary_values = explainer(summary_X)
    shap.summary_plot(summary_values, summary_X, show=False, max_display=25)
    plt.tight_layout()
    summary_path = output_dir / f"{args.target}_lv3_shap_summary.png"
    plt.savefig(summary_path, dpi=180, bbox_inches="tight")
    plt.close()

    explain_date = pd.Timestamp(args.date)
    date_mask = pd.to_datetime(joined["date"]).dt.normalize().eq(explain_date.normalize())
    if not date_mask.any():
        raise ValueError(f"No training row found for explanation date {args.date}")
    row_pos = int(np.flatnonzero(date_mask.to_numpy())[0])
    row_X = features.iloc[[row_pos]]
    row_values = explainer(row_X)
    shap.plots.waterfall(row_values[0], show=False, max_display=25)
    plt.tight_layout()
    waterfall_path = output_dir / f"{args.target}_lv3_shap_waterfall_{explain_date:%Y%m%d}.png"
    plt.savefig(waterfall_path, dpi=180, bbox_inches="tight")
    plt.close()
    contribution_path = output_dir / f"{args.target}_lv3_shap_waterfall_{explain_date:%Y%m%d}.csv"
    _save_feature_contribution_table(row_values, row_X, 0, contribution_path)

    print(f"saved {summary_path}")
    print(f"saved {waterfall_path}")
    print(f"saved {contribution_path}")
    print(f"saved {importance_path}")


if __name__ == "__main__":
    main()
