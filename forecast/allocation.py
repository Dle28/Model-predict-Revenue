from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .common import EPS, safe_div


def historical_weight_tables(
    daily: pd.DataFrame,
    target_col: str,
    end_date: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hist = daily[(daily["date"] <= end_date) & daily[target_col].notna()].copy()
    week_sum = hist.groupby("week_id")[target_col].transform("sum")
    hist = hist[week_sum > EPS].copy()
    hist["actual_weight"] = hist[target_col] / week_sum[week_sum > EPS]
    month_weekday = hist.groupby(["month", "iso_weekday"], as_index=False)["actual_weight"].mean()
    weekday = hist.groupby(["iso_weekday"], as_index=False)["actual_weight"].mean()
    baseline = weekday.rename(columns={"actual_weight": "weekday_baseline"})

    adjustment_rows = []
    adjustment_features = [
        "active_promo_count",
        "is_payday_window",
        "is_month_start",
        "is_month_end",
    ]
    for feature in adjustment_features:
        if feature not in hist.columns:
            continue
        tmp = hist[["iso_weekday", "actual_weight", feature]].copy()
        tmp["feature"] = feature
        tmp["feature_value"] = (tmp[feature] > 0).astype(int)
        effect = tmp.groupby(["feature", "feature_value", "iso_weekday"], as_index=False)["actual_weight"].mean()
        effect = effect.merge(baseline, on="iso_weekday", how="left")
        effect["multiplier"] = safe_div(effect["actual_weight"], effect["weekday_baseline"]).clip(lower=0.75, upper=1.35)
        adjustment_rows.append(effect[["feature", "feature_value", "iso_weekday", "multiplier"]])
    adjustments = pd.concat(adjustment_rows, ignore_index=True) if adjustment_rows else pd.DataFrame(
        columns=["feature", "feature_value", "iso_weekday", "multiplier"]
    )
    return month_weekday, weekday, adjustments


def attach_weights(
    future_daily: pd.DataFrame,
    month_weekday: pd.DataFrame,
    weekday: pd.DataFrame,
    adjustments: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    out = future_daily.merge(month_weekday, on=["month", "iso_weekday"], how="left")
    out = out.rename(columns={"actual_weight": "weight"})
    out = out.merge(weekday.rename(columns={"actual_weight": "weekday_weight"}), on="iso_weekday", how="left")
    out["weight"] = out["weight"].fillna(out["weekday_weight"]).fillna(1.0 / 7.0)
    if adjustments is not None and len(adjustments):
        for feature in adjustments["feature"].drop_duplicates():
            if feature not in out.columns:
                continue
            adj = adjustments[adjustments["feature"].eq(feature)].drop(columns=["feature"])
            out["_feature_value"] = (out[feature] > 0).astype(int)
            out = out.merge(
                adj,
                left_on=["_feature_value", "iso_weekday"],
                right_on=["feature_value", "iso_weekday"],
                how="left",
            )
            out["weight"] = out["weight"] * out["multiplier"].fillna(1.0)
            out = out.drop(columns=["_feature_value", "feature_value", "multiplier"])
    out["weight"] = out["weight"].clip(lower=0.0)
    denom = out.groupby("week_id")["weight"].transform("sum")
    out["weight"] = np.where(denom > EPS, out["weight"] / denom, 1.0 / out.groupby("week_id")["date"].transform("count"))
    return out.drop(columns=["weekday_weight"])


def allocate_and_reconcile(
    future_daily: pd.DataFrame,
    weekly_pred: pd.DataFrame,
    pred_col: str,
    target_output_col: str,
    month_weekday: pd.DataFrame,
    weekday: pd.DataFrame,
    adjustments: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    out = attach_weights(future_daily, month_weekday, weekday, adjustments)
    out = out.merge(weekly_pred[["week_id", pred_col]], on="week_id", how="left")
    out[target_output_col] = out[pred_col].fillna(0.0) * out["weight"]

    sums = out.groupby("week_id")[target_output_col].transform("sum")
    targets = out[pred_col].fillna(0.0)
    out[target_output_col] = np.where(sums > EPS, out[target_output_col] * targets / sums, 0.0)
    out[target_output_col] = out[target_output_col].clip(lower=0.0)
    return out[["date", "week_id", "week_start", target_output_col]]


def daily_seasonal_shim(sales_daily: pd.DataFrame, dates: pd.Series, target_col: str) -> pd.Series:
    hist = sales_daily.copy()
    hist["year"] = hist["date"].dt.year
    hist["month"] = hist["date"].dt.month
    hist["day"] = hist["date"].dt.day

    full_years = hist[(hist["year"] >= 2013) & (hist["year"] <= 2022)]
    annual = full_years.groupby("year")[target_col].sum()
    yoy = annual.pct_change().dropna()
    growth = float((1.0 + yoy).prod() ** (1.0 / len(yoy))) if len(yoy) else 1.0
    base = float(annual.loc[2022] / 365.0) if 2022 in annual.index else float(hist[target_col].mean())

    year_mean = hist.groupby("year")[target_col].transform("mean")
    hist["norm"] = hist[target_col] / np.maximum(year_mean, EPS)
    seasonal = hist.groupby(["month", "day"])["norm"].mean()

    out = []
    for dt in pd.to_datetime(dates):
        key = (int(dt.month), int(dt.day))
        norm = float(seasonal.get(key, 1.0))
        years_ahead = int(dt.year) - 2022
        out.append(max(base * (growth**years_ahead) * norm, 0.0))
    return pd.Series(out, index=dates.index, dtype=float)


ALLOCATION_FEATURE_COLUMNS = [
    "date",
    "week_id",
    "week_start",
    "month",
    "iso_weekday",
    "active_promo_count",
    "is_payday_window",
    "is_month_start",
    "is_month_end",
]


def allocation_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in ALLOCATION_FEATURE_COLUMNS if col in df.columns]
