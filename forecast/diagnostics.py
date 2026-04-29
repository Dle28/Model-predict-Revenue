from __future__ import annotations

import numpy as np
import pandas as pd

from .common import safe_div


def yearly_revenue_diagnostics(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.loc[daily["revenue"].notna(), ["date", "revenue"]].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "total_revenue",
                "mean_daily_revenue",
                "median_daily_revenue",
                "p75_daily_revenue",
                "p90_daily_revenue",
                "mean_4w_rolling",
                "peak_4w_rolling",
            ]
        )
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date")
    frame["year"] = frame["date"].dt.year
    frame["revenue_4w_rolling"] = frame["revenue"].rolling(28, min_periods=14).mean()
    out = frame.groupby("year", as_index=False).agg(
        total_revenue=("revenue", "sum"),
        mean_daily_revenue=("revenue", "mean"),
        median_daily_revenue=("revenue", "median"),
        p75_daily_revenue=("revenue", lambda s: float(np.nanquantile(s, 0.75))),
        p90_daily_revenue=("revenue", lambda s: float(np.nanquantile(s, 0.90))),
        mean_4w_rolling=("revenue_4w_rolling", "mean"),
        peak_4w_rolling=("revenue_4w_rolling", "max"),
    )
    return out.round(2)


def recovery_anchor_diagnostic(weekly_pred: pd.DataFrame) -> pd.DataFrame:
    if weekly_pred.empty:
        return pd.DataFrame()
    out = weekly_pred[["week_id", "week_start"]].copy()
    for target in ["revenue", "cogs"]:
        base_col = f"{target}_w_base"
        final_col = f"{target}_w_pred"
        baseline_col = f"{target}_w_pre_covid_baseline_same_week"
        anchor_col = f"{target}_w_recovery_anchor"
        progress_col = f"{target}_w_recovery_progress"
        recovery_weight_col = f"{target}_w_recovery_weight"
        same_iso_weight_col = f"{target}_w_same_iso_weight"
        model_weight_col = f"{target}_w_lv1_model_weight"

        for col in [
            base_col,
            final_col,
            baseline_col,
            anchor_col,
            progress_col,
            model_weight_col,
            same_iso_weight_col,
            recovery_weight_col,
        ]:
            if col in weekly_pred.columns:
                out[col] = weekly_pred[col]

        if base_col in out.columns and anchor_col in out.columns:
            out[f"{target}_lv1_base_to_recovery_anchor_ratio"] = safe_div(out[base_col], out[anchor_col]).replace(
                [np.inf, -np.inf],
                np.nan,
            )
        if base_col in out.columns and baseline_col in out.columns:
            out[f"{target}_lv1_base_to_precovid_baseline_ratio"] = safe_div(out[base_col], out[baseline_col]).replace(
                [np.inf, -np.inf],
                np.nan,
            )
        if final_col in out.columns and base_col in out.columns:
            out[f"{target}_final_to_lv1_base_ratio"] = safe_div(out[final_col], out[base_col]).replace(
                [np.inf, -np.inf],
                np.nan,
            )

    numeric_cols = [col for col in out.columns if col not in {"week_id", "week_start"}]
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return out


def recovery_anchor_summary(recovery_diag: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in ["revenue", "cogs"]:
        base_col = f"{target}_w_base"
        anchor_col = f"{target}_w_recovery_anchor"
        baseline_col = f"{target}_w_pre_covid_baseline_same_week"
        progress_col = f"{target}_w_recovery_progress"
        if base_col not in recovery_diag.columns:
            continue
        ratio_anchor = (
            safe_div(recovery_diag[base_col], recovery_diag[anchor_col]).replace([np.inf, -np.inf], np.nan)
            if anchor_col in recovery_diag.columns
            else pd.Series(np.nan, index=recovery_diag.index)
        )
        ratio_baseline = (
            safe_div(recovery_diag[base_col], recovery_diag[baseline_col]).replace([np.inf, -np.inf], np.nan)
            if baseline_col in recovery_diag.columns
            else pd.Series(np.nan, index=recovery_diag.index)
        )
        rows.append(
            {
                "target": target,
                "weeks": int(len(recovery_diag)),
                "avg_recovery_progress": float(recovery_diag[progress_col].mean()) if progress_col in recovery_diag.columns else np.nan,
                "p10_base_to_anchor_ratio": float(ratio_anchor.quantile(0.10)) if ratio_anchor.notna().any() else np.nan,
                "median_base_to_anchor_ratio": float(ratio_anchor.median()) if ratio_anchor.notna().any() else np.nan,
                "p10_base_to_baseline_ratio": float(ratio_baseline.quantile(0.10)) if ratio_baseline.notna().any() else np.nan,
                "median_base_to_baseline_ratio": float(ratio_baseline.median()) if ratio_baseline.notna().any() else np.nan,
                "min_lv1_base": float(recovery_diag[base_col].min()),
                "max_lv1_base": float(recovery_diag[base_col].max()),
            }
        )
    return pd.DataFrame(rows).round(4)
