from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .common import EPS, covid_allocation_regime, covid_regime_flags, safe_div
from .lv1 import RidgeLogModel


ALLOCATION_BLEND_CANDIDATES = [0.0, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]
ALLOCATION_TUNING_WEEKS = 13

LV2_FEATURE_COLUMNS = [
    "date",
    "week_id",
    "week_start",
    "month",
    "quarter",
    "iso_weekday",
    "is_weekend",
    "day_of_month",
    "day_of_year",
    "is_payday_window",
    "week_position_in_month",
    "is_month_start",
    "is_month_end",
    "is_holiday",
    "is_tet_window",
    "is_black_friday_window",
    "is_double_day_sale",
    "is_1111_1212",
    "is_year_end_window",
    "is_new_year_window",
    "active_promo_count",
    "avg_promo_discount_value",
    "stackable_promo_count",
    "promo_start_day",
    "promo_end_day",
    "promo_intensity_day",
    "days_to_promo",
    "days_since_promo_start",
    "days_until_promo_end",
]


@dataclass
class AllocationModel:
    target_col: str
    model: Optional[RidgeLogModel]
    month_weekday: pd.DataFrame
    weekday: pd.DataFrame
    day_of_month: pd.DataFrame
    weekday_regime: pd.DataFrame
    legacy_month_weekday: pd.DataFrame
    legacy_weekday: pd.DataFrame
    legacy_adjustments: pd.DataFrame
    feature_columns: List[str]
    fallback_weight: float = 1.0 / 7.0
    score_blend_weight: float = 0.0
    validation_weight_mae: float = np.nan


def lv2_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in LV2_FEATURE_COLUMNS if col in df.columns]


def normalize_weekly_weights(df: pd.DataFrame, weight_col: str = "weight") -> pd.Series:
    clipped = df[weight_col].clip(lower=0.0)
    denom = clipped.groupby(df["week_id"]).transform("sum")
    fallback = 1.0 / df.groupby("week_id")["date"].transform("count")
    return pd.Series(np.where(denom > EPS, clipped / denom, fallback), index=df.index, dtype=float)


def historical_weight_tables(
    daily: pd.DataFrame,
    target_col: str,
    end_date: pd.Timestamp,
    start_date: pd.Timestamp | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hist = daily[(daily["date"] <= end_date) & daily[target_col].notna()].copy()
    if start_date is not None:
        hist = hist[hist["date"] >= start_date].copy()
    if hist.empty:
        weekday = pd.DataFrame({"iso_weekday": range(1, 8), "actual_weight": [1.0 / 7.0] * 7})
        return pd.DataFrame(columns=["month", "iso_weekday", "actual_weight"]), weekday, pd.DataFrame()

    complete_week_counts = hist.groupby("week_id")[target_col].transform("count")
    hist = hist[complete_week_counts.eq(7)].copy()
    week_sum = hist.groupby("week_id")[target_col].transform("sum")
    hist = hist[week_sum > EPS].copy()
    hist["actual_weight"] = hist[target_col] / week_sum[week_sum > EPS]

    month_weekday = hist.groupby(["month", "iso_weekday"], as_index=False)["actual_weight"].mean()
    weekday = hist.groupby(["iso_weekday"], as_index=False)["actual_weight"].mean()
    if weekday.empty:
        weekday = pd.DataFrame({"iso_weekday": range(1, 8), "actual_weight": [1.0 / 7.0] * 7})
    baseline = weekday.rename(columns={"actual_weight": "weekday_baseline"})
    adjustment_rows = []
    adjustment_features = [
        "is_payday_window",
        "is_month_start",
        "is_month_end",
        "is_holiday",
        "active_promo_count",
    ]
    hist["_promo_near"] = hist.get("days_to_promo", 999.0).fillna(999.0).le(3).astype(int)
    adjustment_features.append("_promo_near")
    for feature in adjustment_features:
        if feature not in hist.columns:
            continue
        tmp = hist[["iso_weekday", "actual_weight", feature]].copy()
        tmp["feature"] = "promo_near" if feature == "_promo_near" else feature
        tmp["feature_value"] = tmp[feature].fillna(0).gt(0).astype(int)
        effect = tmp.groupby(["feature", "feature_value", "iso_weekday"], as_index=False)["actual_weight"].mean()
        effect = effect.merge(baseline, on="iso_weekday", how="left")
        effect["multiplier"] = safe_div(effect["actual_weight"], effect["weekday_baseline"]).clip(lower=0.70, upper=1.50)
        adjustment_rows.append(effect[["feature", "feature_value", "iso_weekday", "multiplier"]])
    adjustments = pd.concat(adjustment_rows, ignore_index=True) if adjustment_rows else pd.DataFrame(
        columns=["feature", "feature_value", "iso_weekday", "multiplier"]
    )
    return month_weekday, weekday, adjustments


def attach_base_weights(
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
            if feature == "promo_near":
                feature_values = out.get("days_to_promo", pd.Series(999.0, index=out.index)).fillna(999.0).le(3).astype(int)
            elif feature in out.columns:
                feature_values = out[feature].fillna(0).gt(0).astype(int)
            else:
                continue
            adj = adjustments[adjustments["feature"].eq(feature)].drop(columns=["feature"])
            out["_feature_value"] = feature_values
            out = out.merge(
                adj,
                left_on=["_feature_value", "iso_weekday"],
                right_on=["feature_value", "iso_weekday"],
                how="left",
            )
            out["weight"] = out["weight"] * out["multiplier"].fillna(1.0)
            out = out.drop(columns=["_feature_value", "feature_value", "multiplier"])
    out["weight"] = normalize_weekly_weights(out, "weight")
    return out.drop(columns=["weekday_weight"])


def _allocation_regime(week_id: pd.Series) -> pd.Series:
    return covid_allocation_regime(week_id.astype(str))


def _attach_regime_progress(out: pd.DataFrame) -> pd.DataFrame:
    week_start = pd.to_datetime(out["week_start"]) if "week_start" in out.columns else pd.to_datetime(out["date"])
    flags = [covid_regime_flags(str(wid), wstart) for wid, wstart in zip(out["week_id"].astype(str), week_start)]
    flag_frame = pd.DataFrame(flags, index=out.index)
    for col in flag_frame.columns:
        out[col] = flag_frame[col].astype(float)
    return out


def _historical_allocation_tables(hist: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if hist.empty:
        weekday = pd.DataFrame({"iso_weekday": range(1, 8), "hist_weight_weekday": [1.0 / 7.0] * 7})
        return (
            pd.DataFrame(columns=["month", "iso_weekday", "hist_weight_month_weekday"]),
            weekday,
            pd.DataFrame(columns=["day_of_month", "hist_weight_day_of_month"]),
            pd.DataFrame(columns=["allocation_regime", "iso_weekday", "hist_weight_weekday_regime"]),
        )
    month_weekday = hist.groupby(["month", "iso_weekday"], as_index=False)["actual_weight"].mean()
    month_weekday = month_weekday.rename(columns={"actual_weight": "hist_weight_month_weekday"})
    weekday = hist.groupby("iso_weekday", as_index=False)["actual_weight"].mean()
    weekday = weekday.rename(columns={"actual_weight": "hist_weight_weekday"})
    day_of_month = hist.groupby("day_of_month", as_index=False)["actual_weight"].mean()
    day_of_month = day_of_month.rename(columns={"actual_weight": "hist_weight_day_of_month"})
    weekday_regime = hist.groupby(["allocation_regime", "iso_weekday"], as_index=False)["actual_weight"].mean()
    weekday_regime = weekday_regime.rename(columns={"actual_weight": "hist_weight_weekday_regime"})
    return month_weekday, weekday, day_of_month, weekday_regime


def _attach_historical_weight_features(df: pd.DataFrame, model: AllocationModel) -> pd.DataFrame:
    out = df.copy()
    out["allocation_regime"] = _allocation_regime(out["week_id"].astype(str))
    out = _attach_regime_progress(out)
    out = out.merge(model.month_weekday, on=["month", "iso_weekday"], how="left")
    out = out.merge(model.weekday, on="iso_weekday", how="left")
    out = out.merge(model.day_of_month, on="day_of_month", how="left")
    out = out.merge(model.weekday_regime, on=["allocation_regime", "iso_weekday"], how="left")
    if len(model.weekday_regime):
        regime_wide = model.weekday_regime.pivot_table(
            index="iso_weekday",
            columns="allocation_regime",
            values="hist_weight_weekday_regime",
            aggfunc="mean",
        ).reset_index()
        regime_wide.columns.name = None
        rename = {
            "pre_covid": "hist_weight_weekday_precovid",
            "covid_drop": "hist_weight_weekday_covid",
            "recovery_phase": "hist_weight_weekday_recovery",
            "normalization_phase": "hist_weight_weekday_normal",
        }
        regime_wide = regime_wide.rename(columns=rename)
        out = out.merge(regime_wide, on="iso_weekday", how="left")
    for col in [
        "hist_weight_weekday_precovid",
        "hist_weight_weekday_covid",
        "hist_weight_weekday_recovery",
        "hist_weight_weekday_normal",
    ]:
        if col not in out.columns:
            out[col] = np.nan
    out["hist_weight_weekday"] = out["hist_weight_weekday"].fillna(model.fallback_weight)
    for col in [
        "hist_weight_weekday_precovid",
        "hist_weight_weekday_covid",
        "hist_weight_weekday_recovery",
        "hist_weight_weekday_normal",
    ]:
        out[col] = out[col].fillna(out["hist_weight_weekday"])
    out["hist_weight_month_weekday"] = out["hist_weight_month_weekday"].fillna(out["hist_weight_weekday"])
    out["hist_weight_day_of_month"] = out["hist_weight_day_of_month"].fillna(out["hist_weight_weekday"])
    out["hist_weight_weekday_regime"] = out["hist_weight_weekday_regime"].fillna(out["hist_weight_weekday"])
    recovery_prior = (
        (1.0 - out["recovery_progress"].clip(0.0, 1.0)) * out["hist_weight_weekday_covid"]
        + out["recovery_progress"].clip(0.0, 1.0) * out["hist_weight_weekday_precovid"]
    )
    recovery_like = out["recovery_phase"].gt(0) | out["normalization_phase"].gt(0)
    out["hist_weight_weekday_regime"] = np.where(recovery_like, recovery_prior, out["hist_weight_weekday_regime"])
    out["hist_weight_blend"] = (
        0.45 * out["hist_weight_month_weekday"]
        + 0.35 * out["hist_weight_weekday_regime"]
        + 0.15 * out["hist_weight_day_of_month"]
        + 0.05 * out["hist_weight_weekday"]
    )
    out["hist_weight_blend"] = normalize_weekly_weights(out, "hist_weight_blend")
    if len(model.legacy_weekday):
        legacy = attach_base_weights(
            out[lv2_columns(out)].copy(),
            model.legacy_month_weekday,
            model.legacy_weekday,
            model.legacy_adjustments,
        )[["date", "week_id", "weight"]].rename(columns={"weight": "hist_weight_adjusted"})
        out = out.merge(legacy, on=["date", "week_id"], how="left")
        out["hist_weight_adjusted"] = out["hist_weight_adjusted"].fillna(out["hist_weight_blend"])
        out["hist_weight_blend"] = out["hist_weight_adjusted"]
        out["hist_weight_blend"] = normalize_weekly_weights(out, "hist_weight_blend")
    else:
        out["hist_weight_adjusted"] = out["hist_weight_blend"]
    return out


def _weekly_context(weekly_base: pd.DataFrame, base_col: str) -> pd.DataFrame:
    cols = ["week_id", "week_start", base_col]
    out = weekly_base[[c for c in cols if c in weekly_base.columns]].copy()
    out = out.sort_values("week_start" if "week_start" in out.columns else "week_id")
    out = out.rename(columns={base_col: "weekly_base_value"})
    out["weekly_base_value"] = out["weekly_base_value"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    out["log_weekly_base"] = np.log1p(out["weekly_base_value"])
    rolling = out["weekly_base_value"].shift(1).rolling(4, min_periods=1).mean()
    out["weekly_trend_1w"] = safe_div(out["weekly_base_value"] - out["weekly_base_value"].shift(1), out["weekly_base_value"].shift(1))
    out["weekly_vs_ma4"] = safe_div(out["weekly_base_value"], rolling) - 1.0
    out["weekly_volatility_4w"] = safe_div(out["weekly_base_value"].shift(1).rolling(4, min_periods=2).std(), rolling)
    for col in ["weekly_trend_1w", "weekly_vs_ma4"]:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=-5.0, upper=5.0)
    out["weekly_volatility_4w"] = out["weekly_volatility_4w"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=5.0)
    return out.drop(columns=["week_start"], errors="ignore")


def _allocation_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    date = pd.to_datetime(df["date"])
    iso_weekday = df.get("iso_weekday", pd.Series(1, index=df.index)).fillna(1).astype(float)
    month = df.get("month", pd.Series(1, index=df.index)).fillna(1).astype(float)
    day_of_month = df.get("day_of_month", date.dt.day).fillna(date.dt.day).astype(float)
    day_of_year = df.get("day_of_year", date.dt.dayofyear).fillna(date.dt.dayofyear).astype(float)

    out = pd.DataFrame(index=df.index)
    out["bias"] = 1.0
    for weekday in range(1, 8):
        out[f"wday_{weekday}"] = iso_weekday.eq(weekday).astype(float)
    for month_id in range(1, 13):
        out[f"month_{month_id}"] = month.eq(month_id).astype(float)
    out["dom_sin"] = np.sin(2 * np.pi * day_of_month / 31.0)
    out["dom_cos"] = np.cos(2 * np.pi * day_of_month / 31.0)
    out["doy_sin"] = np.sin(2 * np.pi * day_of_year / 366.0)
    out["doy_cos"] = np.cos(2 * np.pi * day_of_year / 366.0)
    out["week_position_in_month"] = df.get("week_position_in_month", np.ceil(day_of_month / 7.0)).astype(float)
    for col in [
        "pre_covid",
        "covid_drop",
        "recovery_phase",
        "normalization_phase",
        "weeks_since_covid_start",
        "weeks_since_recovery_start",
        "recovery_progress",
    ]:
        out[col] = df[col].fillna(0).astype(float) if col in df.columns else 0.0
    out["recovery_progress_x_month_sin"] = out["recovery_progress"] * np.sin(2 * np.pi * month / 12.0)
    out["recovery_progress_x_month_cos"] = out["recovery_progress"] * np.cos(2 * np.pi * month / 12.0)
    for col in [
        "is_weekend",
        "is_payday_window",
        "is_month_start",
        "is_month_end",
        "is_holiday",
        "is_tet_window",
        "is_black_friday_window",
        "is_double_day_sale",
        "is_1111_1212",
        "is_year_end_window",
        "is_new_year_window",
        "promo_start_day",
        "promo_end_day",
    ]:
        out[col] = df[col].fillna(0).astype(float) if col in df.columns else 0.0
    for col in [
        "active_promo_count",
        "avg_promo_discount_value",
        "stackable_promo_count",
        "promo_intensity_day",
        "days_to_promo",
        "days_since_promo_start",
        "days_until_promo_end",
        "hist_weight_weekday",
        "hist_weight_month_weekday",
        "hist_weight_day_of_month",
        "hist_weight_weekday_regime",
        "hist_weight_weekday_precovid",
        "hist_weight_weekday_covid",
        "hist_weight_weekday_recovery",
        "hist_weight_weekday_normal",
        "hist_weight_adjusted",
        "hist_weight_blend",
        "weekly_base_value",
        "log_weekly_base",
        "weekly_trend_1w",
        "weekly_vs_ma4",
        "weekly_volatility_4w",
    ]:
        raw = df[col] if col in df.columns else pd.Series(0.0, index=df.index)
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        if col.startswith("days_"):
            raw = raw.clip(lower=0.0, upper=999.0)
            out[f"log1p_{col}"] = np.log1p(raw)
        else:
            out[col] = raw
    return out


def _softmax_weights(score: np.ndarray, week_id: pd.Series) -> np.ndarray:
    frame = pd.DataFrame({"week_id": week_id.to_numpy(), "score": np.asarray(score, dtype=float)})
    frame["score"] = frame["score"].replace([np.inf, -np.inf], np.nan).fillna(-12.0).clip(lower=-12.0, upper=2.0)
    group_max = frame.groupby("week_id")["score"].transform("max")
    frame["exp_score"] = np.exp(frame["score"] - group_max)
    denom = frame.groupby("week_id")["exp_score"].transform("sum")
    counts = frame.groupby("week_id")["score"].transform("count")
    return np.where(denom > EPS, frame["exp_score"] / denom, 1.0 / counts)


def _allocation_weight_mae(actual_weight: pd.Series, pred_weight: np.ndarray) -> float:
    actual = actual_weight.replace([np.inf, -np.inf], np.nan).astype(float).to_numpy()
    pred = np.asarray(pred_weight, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(pred)
    if not mask.any():
        return float("inf")
    return float(np.mean(np.abs(actual[mask] - pred[mask])))


def _tune_allocation_blend_weight(train: pd.DataFrame, features: pd.DataFrame, target: pd.Series) -> Tuple[float, float]:
    complete_weeks = train[["week_id", "week_start"]].drop_duplicates().sort_values("week_start")
    if len(complete_weeks) < ALLOCATION_TUNING_WEEKS + 8:
        return 0.0, np.nan
    val_weeks = set(complete_weeks.tail(ALLOCATION_TUNING_WEEKS)["week_id"])
    train_mask = ~train["week_id"].isin(val_weeks)
    val_mask = train["week_id"].isin(val_weeks)
    if train_mask.sum() < 30 or val_mask.sum() < 7:
        return 0.0, np.nan

    usable_train = train_mask & target.replace([np.inf, -np.inf], np.nan).notna()
    if usable_train.sum() < 30:
        return 0.0, np.nan
    temp_model = RidgeLogModel(alpha=50.0).fit(features.loc[usable_train], target.loc[usable_train])
    model_score = np.asarray(temp_model.predict(features.loc[val_mask]), dtype=float)
    hist_score = np.log(train.loc[val_mask, "hist_weight_blend"].clip(lower=1e-5).to_numpy())
    actual_weight = train.loc[val_mask, "actual_weight"]

    best_alpha = 0.0
    best_mae = float("inf")
    for alpha in ALLOCATION_BLEND_CANDIDATES:
        score = hist_score + alpha * model_score
        pred_weight = _softmax_weights(score, train.loc[val_mask, "week_id"])
        mae = _allocation_weight_mae(actual_weight, pred_weight)
        if mae < best_mae - 1e-6:
            best_alpha = float(alpha)
            best_mae = mae
    return best_alpha, best_mae


def fit_allocation_model(
    daily: pd.DataFrame,
    target_col: str,
    weekly_base: pd.DataFrame,
    base_col: str,
    train_end_date: pd.Timestamp,
    train_start_date: Optional[pd.Timestamp] = None,
) -> AllocationModel:
    hist = daily[(daily["date"] <= train_end_date) & daily[target_col].notna()].copy()
    if train_start_date is not None:
        hist = hist[hist["date"] >= train_start_date].copy()
    complete_week_counts = hist.groupby("week_id")[target_col].transform("count")
    hist = hist[complete_week_counts.eq(7)].copy()
    week_sum = hist.groupby("week_id")[target_col].transform("sum")
    hist = hist[week_sum > EPS].copy()
    hist["actual_weight"] = hist[target_col] / week_sum[week_sum > EPS]
    hist["allocation_regime"] = _allocation_regime(hist["week_id"].astype(str))

    month_weekday, weekday, day_of_month, weekday_regime = _historical_allocation_tables(hist)
    legacy_month_weekday, legacy_weekday, legacy_adjustments = historical_weight_tables(
        daily,
        target_col,
        train_end_date,
        train_start_date,
    )
    fallback = float(hist["actual_weight"].median()) if len(hist) else 1.0 / 7.0
    allocation_model = AllocationModel(
        target_col=target_col,
        model=None,
        month_weekday=month_weekday,
        weekday=weekday,
        day_of_month=day_of_month,
        weekday_regime=weekday_regime,
        legacy_month_weekday=legacy_month_weekday,
        legacy_weekday=legacy_weekday,
        legacy_adjustments=legacy_adjustments,
        feature_columns=[],
        fallback_weight=fallback if np.isfinite(fallback) and fallback > 0 else 1.0 / 7.0,
    )
    if hist.empty:
        return allocation_model

    train = _attach_historical_weight_features(hist, allocation_model)
    train = train.merge(_weekly_context(weekly_base, base_col), on="week_id", how="left")
    features = _allocation_feature_frame(train)
    target = np.log(
        train["actual_weight"].clip(lower=1e-5)
        / train["hist_weight_blend"].clip(lower=1e-5)
    )
    usable = target.replace([np.inf, -np.inf], np.nan).notna()
    if usable.sum() >= 30:
        best_alpha, best_mae = _tune_allocation_blend_weight(train, features, target)
        allocation_model.model = RidgeLogModel(alpha=35.0).fit(features.loc[usable], target.loc[usable])
        allocation_model.feature_columns = allocation_model.model.features
        allocation_model.score_blend_weight = best_alpha
        allocation_model.validation_weight_mae = best_mae
    return allocation_model


def allocate_base_daily_dynamic(
    future_daily: pd.DataFrame,
    weekly_base: pd.DataFrame,
    base_col: str,
    target_output_col: str,
    model: AllocationModel,
) -> pd.DataFrame:
    out = _attach_historical_weight_features(future_daily, model)
    out = out.merge(_weekly_context(weekly_base, base_col), on="week_id", how="left")
    features = _allocation_feature_frame(out)
    if model.model is None:
        score = np.log(out["hist_weight_blend"].clip(lower=1e-5).to_numpy())
    else:
        model_score = np.asarray(model.model.predict(features), dtype=float)
        hist_score = np.log(out["hist_weight_blend"].clip(lower=1e-5).to_numpy())
        alpha = float(np.clip(model.score_blend_weight, 0.0, 1.0))
        score = hist_score + alpha * model_score
    score = np.clip(score, -12.0, 2.0)
    out["_score"] = score
    group_max = out.groupby("week_id")["_score"].transform("max")
    out["_exp_score"] = np.exp(out["_score"] - group_max)
    denom = out.groupby("week_id")["_exp_score"].transform("sum")
    fallback = out["hist_weight_blend"].clip(lower=0.0)
    fallback_denom = fallback.groupby(out["week_id"]).transform("sum")
    fallback_weight = np.where(fallback_denom > EPS, fallback / fallback_denom, 1.0 / out.groupby("week_id")["date"].transform("count"))
    out["weight"] = np.where(denom > EPS, out["_exp_score"] / denom, fallback_weight)
    baseline_col = base_col.replace("_base", "_pre_covid_baseline_same_week")
    progress_col = base_col.replace("_base", "_recovery_progress")
    weekly_cols = ["week_id", base_col]
    for col in [baseline_col, progress_col]:
        if col in weekly_base.columns:
            weekly_cols.append(col)
    out = out.merge(weekly_base[weekly_cols], on="week_id", how="left")
    out[target_output_col] = out[base_col].fillna(0.0) * out["weight"]
    out[target_output_col] = out[target_output_col].clip(lower=0.0)
    out["weekly_base_value"] = out[base_col].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    extra_cols = ["weekly_base_value"]
    if baseline_col in out.columns:
        out["pre_covid_baseline_same_week"] = out[baseline_col]
        extra_cols.append("pre_covid_baseline_same_week")
    if progress_col in out.columns:
        out["recovery_progress"] = out[progress_col]
        extra_cols.append("recovery_progress")
    return out[["date", "week_id", "week_start", target_output_col, "weight"] + extra_cols]


def allocate_base_daily(
    future_daily: pd.DataFrame,
    weekly_base: pd.DataFrame,
    base_col: str,
    target_output_col: str,
    month_weekday: pd.DataFrame,
    weekday: pd.DataFrame,
    adjustments: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    out = attach_base_weights(future_daily, month_weekday, weekday, adjustments)
    out = out.merge(weekly_base[["week_id", base_col]], on="week_id", how="left")
    out[target_output_col] = out[base_col].fillna(0.0) * out["weight"]
    out[target_output_col] = out[target_output_col].clip(lower=0.0)
    out["weekly_base_value"] = out[base_col].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    return out[["date", "week_id", "week_start", target_output_col, "weight", "weekly_base_value"]]


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
