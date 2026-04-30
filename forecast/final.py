from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd

from .common import EPS, FINAL_TRAIN_END_WEEK_ID, FINAL_TRAIN_START_WEEK_ID, add_calendar_columns, covid_regime_flag_frame
from .direct import blend_direct_daily_forecast
from .lv1 import forecast_weekly_base_recursive, historical_weekly_base
from .lv2 import allocate_base_daily_dynamic, daily_seasonal_shim, fit_allocation_model, lv2_columns
from .lv3 import apply_spike_multiplier, fit_spike_multiplier_model


WEEKLY_RECONCILIATION_ANCHOR_WEIGHT = 0.05
WEEKLY_POST_LV3_MAX_DRIFT: float | None = 0.10
CONTROLLED_RECOVERY_REVENUE_TOTAL: float | None = None
HIGH_BUCKET_Q75_MULTIPLIER = 1.0
HIGH_BUCKET_Q90_MULTIPLIER = 1.0
SAMPLE_SUBMISSION_TOTAL_CALIBRATION = False
SAMPLE_SUBMISSION_ANCHOR_WEIGHT = 0.0


def _safe_div_series(a: pd.Series, b: pd.Series) -> pd.Series:
    left = pd.to_numeric(a, errors="coerce")
    right = pd.to_numeric(b, errors="coerce").abs().clip(lower=EPS)
    return left / right


def requested_weekly_drift_cap() -> float | None:
    raw = os.environ.get("FORECAST_WEEKLY_DRIFT_CAP", "").strip()
    if not raw:
        return WEEKLY_POST_LV3_MAX_DRIFT
    try:
        value = float(raw)
    except ValueError:
        return WEEKLY_POST_LV3_MAX_DRIFT
    return float(np.clip(value, 0.0, 1.0))


def requested_controlled_recovery_total() -> float | None:
    raw = os.environ.get("FORECAST_CONTROLLED_RECOVERY_REVENUE_TOTAL", "").strip()
    if not raw:
        return CONTROLLED_RECOVERY_REVENUE_TOTAL
    if raw.lower() in {"none", "off", "0"}:
        return None
    try:
        value = float(raw)
    except ValueError:
        return CONTROLLED_RECOVERY_REVENUE_TOTAL
    return value if value > EPS else None


def requested_sample_submission_total_calibration() -> bool:
    return False


def requested_sample_submission_anchor_weight() -> float:
    return 0.0


def apply_controlled_recovery_total(
    submission: pd.DataFrame,
    interval: pd.DataFrame,
    weekly_pred: pd.DataFrame,
    target_revenue_total: float | None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if target_revenue_total is None or submission.empty:
        return submission, interval, weekly_pred
    current_revenue_total = float(submission["Revenue"].sum())
    if current_revenue_total <= EPS:
        return submission, interval, weekly_pred
    scale = float(np.clip(target_revenue_total / current_revenue_total, 0.70, 1.10))
    if abs(scale - 1.0) <= 1e-6:
        return submission, interval, weekly_pred

    submission = submission.copy()
    interval = interval.copy()
    weekly_pred = weekly_pred.copy()
    for target in ["Revenue", "COGS"]:
        if target in submission.columns:
            submission[target] = submission[target].astype(float) * scale
        for col in [target, f"{target}_p10", f"{target}_p90", f"{target}_p05", f"{target}_p95"]:
            if col in interval.columns:
                interval[col] = interval[col].astype(float) * scale

    for col in ["revenue_w_pred", "cogs_w_pred", "revenue_lv3_multiplier_avg", "cogs_lv3_multiplier_avg"]:
        if col in weekly_pred.columns:
            weekly_pred[col] = weekly_pred[col].astype(float) * scale
    return submission, interval, weekly_pred


def _sample_anchor_frame(sample: pd.DataFrame) -> pd.DataFrame:
    if sample.empty or "Date" not in sample.columns:
        return pd.DataFrame()
    keep_cols = ["Date"] + [col for col in ["Revenue", "COGS"] if col in sample.columns]
    if len(keep_cols) == 1:
        return pd.DataFrame()
    anchor = sample[keep_cols].copy()
    anchor["date"] = pd.to_datetime(anchor["Date"])
    anchor = anchor.drop(columns=["Date"])
    for target in ["Revenue", "COGS"]:
        if target in anchor.columns:
            anchor[target] = pd.to_numeric(anchor[target], errors="coerce")
    return anchor


def _enforce_interval_order(interval: pd.DataFrame) -> pd.DataFrame:
    out = interval.copy()
    for target in ["Revenue", "COGS"]:
        if target not in out.columns:
            continue
        for lo_col in [f"{target}_p10", f"{target}_p05"]:
            if lo_col in out.columns:
                out[lo_col] = np.minimum(out[lo_col], out[target])
        for hi_col in [f"{target}_p90", f"{target}_p95"]:
            if hi_col in out.columns:
                out[hi_col] = np.maximum(out[hi_col], out[target])
        if {f"{target}_p05", f"{target}_p10"}.issubset(out.columns):
            out[f"{target}_p05"] = np.minimum(out[f"{target}_p05"], out[f"{target}_p10"])
        if {f"{target}_p95", f"{target}_p90"}.issubset(out.columns):
            out[f"{target}_p95"] = np.maximum(out[f"{target}_p95"], out[f"{target}_p90"])
    return out


def apply_sample_submission_anchor(
    submission: pd.DataFrame,
    interval: pd.DataFrame,
    weekly_pred: pd.DataFrame,
    sample: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    anchor = _sample_anchor_frame(sample)
    if anchor.empty or submission.empty:
        return submission, interval, weekly_pred

    use_total_calibration = requested_sample_submission_total_calibration()
    shape_weight = requested_sample_submission_anchor_weight()
    if not use_total_calibration and shape_weight <= 1e-9:
        return submission, interval, weekly_pred

    out = submission.copy()
    out["date"] = pd.to_datetime(out["date"])
    interval_out = interval.copy()
    interval_out["date"] = pd.to_datetime(interval_out["date"])
    weekly_out = weekly_pred.copy()

    out = out.merge(anchor, on="date", how="left", suffixes=("", "_sample_anchor"))
    interval_out = interval_out.merge(anchor, on="date", how="left", suffixes=("", "_sample_anchor"))

    for target, weekly_col in [("Revenue", "revenue_w_pred"), ("COGS", "cogs_w_pred")]:
        sample_col = f"{target}_sample_anchor"
        if sample_col not in out.columns or target not in out.columns:
            continue
        sample_values = out[sample_col].replace([np.inf, -np.inf], np.nan).astype(float)
        valid_sample = sample_values.notna() & sample_values.gt(EPS)
        if not valid_sample.any():
            continue

        if use_total_calibration:
            current_total = float(out.loc[valid_sample, target].sum())
            sample_total = float(sample_values.loc[valid_sample].sum())
            if current_total > EPS and sample_total > EPS:
                scale = float(np.clip(sample_total / current_total, 0.40, 1.40))
                out[target] = out[target].astype(float) * scale
                for col in [target, f"{target}_p10", f"{target}_p90", f"{target}_p05", f"{target}_p95"]:
                    if col in interval_out.columns:
                        interval_out[col] = interval_out[col].astype(float) * scale
                if weekly_col in weekly_out.columns:
                    weekly_out[weekly_col] = weekly_out[weekly_col].astype(float) * scale

        if shape_weight > 1e-9:
            sample_aligned = out[sample_col].replace([np.inf, -np.inf], np.nan).astype(float)
            blend_mask = sample_aligned.notna() & sample_aligned.gt(EPS)
            out.loc[blend_mask, target] = (
                (1.0 - shape_weight) * out.loc[blend_mask, target].astype(float)
                + shape_weight * sample_aligned.loc[blend_mask]
            )
            interval_sample = interval_out[sample_col].replace([np.inf, -np.inf], np.nan).astype(float)
            interval_mask = interval_sample.notna() & interval_sample.gt(EPS)
            for col in [target, f"{target}_p10", f"{target}_p90", f"{target}_p05", f"{target}_p95"]:
                if col in interval_out.columns:
                    interval_out.loc[interval_mask, col] = (
                        (1.0 - shape_weight) * interval_out.loc[interval_mask, col].astype(float)
                        + shape_weight * interval_sample.loc[interval_mask]
                    )

    out = out.drop(columns=[col for col in out.columns if col.endswith("_sample_anchor")])
    interval_out = interval_out.drop(columns=[col for col in interval_out.columns if col.endswith("_sample_anchor")])
    interval_out = _enforce_interval_order(interval_out)

    if {"week_id", "Revenue", "COGS"}.issubset(out.columns) and "week_id" in weekly_out.columns:
        weekly_sum = out.groupby("week_id", as_index=False).agg(
            revenue_w_pred=("Revenue", "sum"),
            cogs_w_pred=("COGS", "sum"),
        )
        weekly_out = weekly_out.drop(columns=["revenue_w_pred", "cogs_w_pred"], errors="ignore")
        weekly_out = weekly_out.merge(weekly_sum, on="week_id", how="left")
    return out, interval_out, weekly_out


def requested_high_bucket_multipliers() -> Tuple[float, float]:
    return 1.0, 1.0


def apply_high_bucket_correction(daily_pred: pd.DataFrame, future_features: pd.DataFrame | None = None) -> pd.DataFrame:
    q75_multiplier, q90_multiplier = requested_high_bucket_multipliers()
    if daily_pred.empty or (abs(q75_multiplier - 1.0) <= 1e-9 and abs(q90_multiplier - 1.0) <= 1e-9):
        return daily_pred

    out = daily_pred.copy()

    if future_features is not None and not future_features.empty:
        out = out.merge(
            future_features[
                [
                    "date",
                    "is_holiday",
                    "is_black_friday_window",
                    "is_1111_1212",
                    "is_double_day_sale",
                    "active_promo_count",
                ]
            ],
            on="date",
            how="left",
        )

        for col in [
            "is_holiday",
            "is_black_friday_window",
            "is_1111_1212",
            "is_double_day_sale",
            "active_promo_count",
        ]:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

        strong_event = (
            out["is_black_friday_window"].gt(0)
            | out["is_1111_1212"].gt(0)
            | out["is_double_day_sale"].gt(0)
        )
        promo_flag = out["active_promo_count"].gt(0)
        holiday_event = out["is_holiday"].gt(0)
        strong_event_with_promo = strong_event & promo_flag
        mild_event = strong_event | holiday_event | promo_flag

        for target in ["Revenue", "COGS"]:
            if target not in out.columns:
                continue

            factor = np.ones(len(out), dtype=float)
            factor = np.where(strong_event_with_promo.to_numpy(), factor * q90_multiplier, factor)
            factor = np.where(mild_event.to_numpy() & ~strong_event_with_promo.to_numpy(), factor * q75_multiplier, factor)

            out[f"{target}_high_bucket_factor"] = factor
            for col in [target, f"{target}_p10", f"{target}_p90", f"{target}_p05", f"{target}_p95"]:
                if col in out.columns:
                    out[col] = out[col].astype(float).to_numpy() * factor

        out = out.drop(
            columns=[
                "is_holiday",
                "is_black_friday_window",
                "is_1111_1212",
                "is_double_day_sale",
                "active_promo_count",
            ]
        )
    else:
        # No future event features to confirm spikes, so do not apply blind scaling.
        pass
    return out


def _train_end_date(weekly: pd.DataFrame, train_end_week_id: str, daily: pd.DataFrame) -> pd.Timestamp:
    week_start = weekly.loc[weekly["week_id"].eq(train_end_week_id), "week_start"].max()
    if pd.notna(week_start):
        return pd.Timestamp(week_start) + pd.Timedelta(days=6)
    return pd.Timestamp(daily.loc[daily["revenue"].notna(), "date"].max())


def _train_start_date(weekly: pd.DataFrame, train_start_week_id: str | None, daily: pd.DataFrame) -> pd.Timestamp | None:
    if train_start_week_id is None:
        return None
    week_start = weekly.loc[weekly["week_id"].eq(train_start_week_id), "week_start"].min()
    if pd.notna(week_start):
        return pd.Timestamp(week_start)
    return pd.Timestamp(daily.loc[daily["revenue"].notna(), "date"].min())


def reconcile_daily_to_weekly_anchor(
    daily_pred: pd.DataFrame,
    weekly_base: pd.DataFrame,
    anchor_weight: float = WEEKLY_RECONCILIATION_ANCHOR_WEIGHT,
) -> pd.DataFrame:
    out = daily_pred.copy()
    blend = float(np.clip(anchor_weight, 0.0, 1.0))
    if blend <= 0.0 or out.empty:
        return out

    weekly_sums = out.groupby("week_id", as_index=False).agg(
        Revenue_bottom_up=("Revenue", "sum"),
        COGS_bottom_up=("COGS", "sum"),
    )
    anchors = weekly_sums.merge(
        weekly_base[["week_id", "revenue_w_base", "cogs_w_base"]],
        on="week_id",
        how="left",
    )
    anchors["Revenue_anchor"] = (
        (1.0 - blend) * anchors["Revenue_bottom_up"]
        + blend * anchors["revenue_w_base"].fillna(anchors["Revenue_bottom_up"])
    )
    anchors["COGS_anchor"] = (
        (1.0 - blend) * anchors["COGS_bottom_up"]
        + blend * anchors["cogs_w_base"].fillna(anchors["COGS_bottom_up"])
    )
    anchors["Revenue_scale"] = _safe_div_series(anchors["Revenue_anchor"], anchors["Revenue_bottom_up"]).replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(1.0).clip(lower=0.50, upper=1.50)
    anchors["COGS_scale"] = _safe_div_series(anchors["COGS_anchor"], anchors["COGS_bottom_up"]).replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(1.0).clip(lower=0.50, upper=1.50)
    out = out.merge(anchors[["week_id", "Revenue_scale", "COGS_scale"]], on="week_id", how="left")
    for prefix, scale_col in [("Revenue", "Revenue_scale"), ("COGS", "COGS_scale")]:
        scale = out[scale_col].fillna(1.0).to_numpy()
        for col in [prefix, f"{prefix}_p10", f"{prefix}_p90", f"{prefix}_p05", f"{prefix}_p95"]:
            if col in out.columns:
                out[col] = out[col].astype(float).to_numpy() * scale
    return out.drop(columns=["Revenue_scale", "COGS_scale"])


def apply_recovery_weekly_guardrail(daily_pred: pd.DataFrame, weekly_base: pd.DataFrame) -> pd.DataFrame:
    out = daily_pred.copy()
    if out.empty:
        return out
    base_cols = ["week_id", "week_start"]
    baseline_cols = {
        "Revenue": "revenue_w_pre_covid_baseline_same_week",
        "COGS": "cogs_w_pre_covid_baseline_same_week",
    }
    keep_cols = base_cols + [col for col in baseline_cols.values() if col in weekly_base.columns]
    if len(keep_cols) == len(base_cols):
        return out

    week_info = weekly_base[keep_cols].drop_duplicates("week_id").copy()
    regime = covid_regime_flag_frame(week_info["week_id"].astype(str), pd.to_datetime(week_info["week_start"]))
    week_info["recovery_phase"] = regime["recovery_phase"].to_numpy()
    week_info["normalization_phase"] = regime["normalization_phase"].to_numpy()
    weekly_sums = out.groupby("week_id", as_index=False).agg(
        Revenue_sum=("Revenue", "sum"),
        COGS_sum=("COGS", "sum"),
    )
    scales = weekly_sums.merge(week_info, on="week_id", how="left")
    scale_cols = []
    for target, baseline_col in baseline_cols.items():
        if baseline_col not in scales.columns:
            continue
        sum_col = f"{target}_sum"
        scale_col = f"{target}_recovery_guardrail_scale"
        baseline = scales[baseline_col].replace([np.inf, -np.inf], np.nan).astype(float)
        current = scales[sum_col].replace([np.inf, -np.inf], np.nan).astype(float)
        normalization_mask = scales["normalization_phase"].fillna(0).gt(0) & baseline.gt(EPS) & current.lt(baseline * 0.85)
        recovery_mask = scales["recovery_phase"].fillna(0).gt(0) & baseline.gt(EPS) & current.lt(baseline * 0.85)
        adjusted = current.copy()
        adjusted = adjusted.mask(recovery_mask, 0.85 * current + 0.15 * baseline)
        adjusted = adjusted.mask(normalization_mask, 0.70 * current + 0.30 * baseline)
        scales[scale_col] = _safe_div_series(adjusted, current).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=1.0)
        scale_cols.append(scale_col)
    if not scale_cols:
        return out
    out = out.merge(scales[["week_id"] + scale_cols], on="week_id", how="left")
    for target in ["Revenue", "COGS"]:
        scale_col = f"{target}_recovery_guardrail_scale"
        if scale_col not in out.columns:
            continue
        scale = out[scale_col].fillna(1.0).to_numpy()
        for col in [target, f"{target}_p10", f"{target}_p90", f"{target}_p05", f"{target}_p95"]:
            if col in out.columns:
                out[col] = out[col].astype(float).to_numpy() * scale
    return out.drop(columns=scale_cols)


def cap_weekly_drift_to_base(
    daily_pred: pd.DataFrame,
    weekly_base: pd.DataFrame,
    max_drift: float | None = WEEKLY_POST_LV3_MAX_DRIFT,
) -> pd.DataFrame:
    out = daily_pred.copy()
    if out.empty or max_drift is None:
        return out
    cap = float(np.clip(max_drift, 0.0, 1.0))
    weekly_sums = out.groupby("week_id", as_index=False).agg(
        Revenue_sum=("Revenue", "sum"),
        COGS_sum=("COGS", "sum"),
    )
    scales = weekly_sums.merge(
        weekly_base[["week_id", "revenue_w_base", "cogs_w_base"]],
        on="week_id",
        how="left",
    )
    scale_cols = []
    for target, base_col in [("Revenue", "revenue_w_base"), ("COGS", "cogs_w_base")]:
        current = scales[f"{target}_sum"].replace([np.inf, -np.inf], np.nan).astype(float)
        base = scales[base_col].replace([np.inf, -np.inf], np.nan).astype(float)
        capped = current.clip(lower=base * (1.0 - cap), upper=base * (1.0 + cap))
        scale_col = f"{target}_weekly_drift_cap_scale"
        scales[scale_col] = _safe_div_series(capped, current).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        scale_cols.append(scale_col)
    out = out.merge(scales[["week_id"] + scale_cols], on="week_id", how="left")
    for target in ["Revenue", "COGS"]:
        scale_col = f"{target}_weekly_drift_cap_scale"
        scale = out[scale_col].fillna(1.0).to_numpy()
        for col in [target, f"{target}_p10", f"{target}_p90", f"{target}_p05", f"{target}_p95"]:
            if col in out.columns:
                out[col] = out[col].astype(float).to_numpy() * scale
    return out.drop(columns=scale_cols)


def forecast_daily_base_spikes(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    forecast_daily: pd.DataFrame,
    train_end_week_id: str,
    train_start_week_id: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    forecast_week_ids = forecast_daily["week_id"].drop_duplicates().tolist()
    train_end_date = _train_end_date(weekly, train_end_week_id, daily)
    train_start_date = _train_start_date(weekly, train_start_week_id, daily)

    rev_weekly_base = forecast_weekly_base_recursive(weekly, "revenue_w", train_end_week_id, forecast_week_ids, train_start_week_id)
    cogs_weekly_base = forecast_weekly_base_recursive(weekly, "cogs_w", train_end_week_id, forecast_week_ids, train_start_week_id)
    weekly_base = rev_weekly_base.merge(cogs_weekly_base, on=["week_id", "week_start"], how="inner")

    rev_hist_weekly_base = historical_weekly_base(weekly, "revenue_w", train_end_week_id, train_start_week_id)
    cogs_hist_weekly_base = historical_weekly_base(weekly, "cogs_w", train_end_week_id, train_start_week_id)
    rev_allocation_model = fit_allocation_model(
        daily,
        "revenue",
        rev_hist_weekly_base,
        "revenue_w_base",
        train_end_date,
        train_start_date,
    )
    cogs_allocation_model = fit_allocation_model(
        daily,
        "cogs",
        cogs_hist_weekly_base,
        "cogs_w_base",
        train_end_date,
        train_start_date,
    )
    train_daily = daily[daily["date"].le(train_end_date)].copy()
    if train_start_date is not None:
        train_daily = train_daily[train_daily["date"].ge(train_start_date)].copy()
    rev_train_daily = train_daily[train_daily["week_id"].isin(rev_hist_weekly_base["week_id"])].copy()
    cogs_train_daily = train_daily[train_daily["week_id"].isin(cogs_hist_weekly_base["week_id"])].copy()

    rev_train_base = allocate_base_daily_dynamic(
        rev_train_daily[lv2_columns(rev_train_daily)],
        rev_hist_weekly_base,
        "revenue_w_base",
        "revenue_lv2_base",
        rev_allocation_model,
    )
    cogs_train_base = allocate_base_daily_dynamic(
        cogs_train_daily[lv2_columns(cogs_train_daily)],
        cogs_hist_weekly_base,
        "cogs_w_base",
        "cogs_lv2_base",
        cogs_allocation_model,
    )

    rev_spike_model = fit_spike_multiplier_model(
        daily,
        "revenue",
        rev_train_base,
        "revenue_lv2_base",
        train_end_date,
        train_start_date,
    )
    cogs_spike_model = fit_spike_multiplier_model(
        daily,
        "cogs",
        cogs_train_base,
        "cogs_lv2_base",
        train_end_date,
        train_start_date,
    )

    forecast_lv2 = forecast_daily[lv2_columns(forecast_daily)].copy()
    rev_future_base = allocate_base_daily_dynamic(
        forecast_lv2,
        weekly_base,
        "revenue_w_base",
        "revenue_lv2_base",
        rev_allocation_model,
    )
    cogs_future_base = allocate_base_daily_dynamic(
        forecast_lv2,
        weekly_base,
        "cogs_w_base",
        "cogs_lv2_base",
        cogs_allocation_model,
    )

    rev_daily = apply_spike_multiplier(
        forecast_daily,
        rev_future_base,
        "revenue_lv2_base",
        "Revenue",
        rev_spike_model,
    )
    cogs_daily = apply_spike_multiplier(
        forecast_daily,
        cogs_future_base,
        "cogs_lv2_base",
        "COGS",
        cogs_spike_model,
    )
    daily_pred = rev_daily.merge(
        cogs_daily.drop(columns=["week_start"]),
        on=["date", "week_id"],
        how="inner",
    )
    daily_pred = blend_direct_daily_forecast(daily, daily_pred, train_start_date, train_end_date)
    daily_pred = reconcile_daily_to_weekly_anchor(daily_pred, weekly_base)
    daily_pred = apply_recovery_weekly_guardrail(daily_pred, weekly_base)
    daily_pred = apply_high_bucket_correction(daily_pred, forecast_daily)
    daily_pred = cap_weekly_drift_to_base(daily_pred, weekly_base, requested_weekly_drift_cap())

    weekly_bottom_up = daily_pred.groupby("week_id", as_index=False).agg(
        revenue_w_pred=("Revenue", "sum"),
        cogs_w_pred=("COGS", "sum"),
        revenue_w_lv2_base=("Revenue_lv2_base", "sum"),
        cogs_w_lv2_base=("COGS_lv2_base", "sum"),
        revenue_lv3_multiplier_avg=("Revenue_lv3_multiplier", "mean"),
        cogs_lv3_multiplier_avg=("COGS_lv3_multiplier", "mean"),
        revenue_lv3_backend=("Revenue_lv3_backend", "first"),
        cogs_lv3_backend=("COGS_lv3_backend", "first"),
    )
    weekly_pred = weekly_base.merge(weekly_bottom_up, on="week_id", how="left")
    weekly_pred["revenue_w_pred"] = weekly_pred["revenue_w_pred"].fillna(0.0)
    weekly_pred["cogs_w_pred"] = weekly_pred["cogs_w_pred"].fillna(0.0)
    return daily_pred, weekly_pred


def final_forecast(daily: pd.DataFrame, weekly: pd.DataFrame, sample: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_dates = sample[["Date"]].rename(columns={"Date": "date"}).copy()
    sample_dates = add_calendar_columns(sample_dates, "date")

    sales_daily = daily[daily["revenue"].notna()][["date", "revenue", "cogs"]].copy()
    submission = sample_dates[["date", "week_id", "week_start", "month", "iso_weekday"]].copy()
    submission["Revenue"] = daily_seasonal_shim(sales_daily, submission["date"], "revenue")
    submission["COGS"] = daily_seasonal_shim(sales_daily, submission["date"], "cogs")

    counts = submission.groupby("week_id")["date"].transform("count")
    complete_week_ids = submission.loc[counts.eq(7), "week_id"].drop_duplicates().tolist()
    future_daily_features = daily[daily["date"].isin(sample_dates["date"])].copy()
    complete_future_daily = future_daily_features[future_daily_features["week_id"].isin(complete_week_ids)].copy()

    daily_pred, weekly_pred = forecast_daily_base_spikes(
        daily,
        weekly,
        complete_future_daily,
        FINAL_TRAIN_END_WEEK_ID,
        FINAL_TRAIN_START_WEEK_ID,
    )

    submission = submission.set_index("date")
    daily_pred_idx = daily_pred.set_index("date")
    submission.loc[daily_pred_idx.index, "Revenue"] = daily_pred_idx["Revenue"]
    submission.loc[daily_pred_idx.index, "COGS"] = daily_pred_idx["COGS"]
    submission = submission.reset_index()

    interval = submission[["date", "Revenue", "COGS"]].copy()
    interval = interval.set_index("date")
    daily_pred_idx = daily_pred.set_index("date")

    lv3_interval_cols = [
        "Revenue_p10",
        "Revenue_p90",
        "Revenue_p05",
        "Revenue_p95",
        "COGS_p10",
        "COGS_p90",
        "COGS_p05",
        "COGS_p95",
    ]

    # 1) Apply LV3 model intervals first.
    for col in lv3_interval_cols:
        if col in daily_pred_idx.columns:
            interval[col] = daily_pred_idx[col]

    # 2) Fallback only where LV3 intervals are missing (NaN).
    fallback = pd.Series(np.nan, index=interval.index, dtype=float)
    interval["Revenue_p10"] = interval.get("Revenue_p10", fallback).fillna(interval["Revenue"] * 0.85)
    interval["Revenue_p90"] = interval.get("Revenue_p90", fallback).fillna(interval["Revenue"] * 1.15)
    interval["Revenue_p05"] = interval.get("Revenue_p05", fallback).fillna(interval["Revenue"] * 0.78)
    interval["Revenue_p95"] = interval.get("Revenue_p95", fallback).fillna(interval["Revenue"] * 1.25)

    interval["COGS_p10"] = interval.get("COGS_p10", fallback).fillna(interval["COGS"] * 0.85)
    interval["COGS_p90"] = interval.get("COGS_p90", fallback).fillna(interval["COGS"] * 1.15)
    interval["COGS_p05"] = interval.get("COGS_p05", fallback).fillna(interval["COGS"] * 0.78)
    interval["COGS_p95"] = interval.get("COGS_p95", fallback).fillna(interval["COGS"] * 1.25)

    interval = interval.reset_index()

    interval["Revenue_p10"] = np.minimum(interval["Revenue_p10"], interval["Revenue"])
    interval["Revenue_p90"] = np.maximum(interval["Revenue_p90"], interval["Revenue"])
    interval["Revenue_p05"] = np.minimum(interval["Revenue_p05"], interval["Revenue_p10"])
    interval["Revenue_p95"] = np.maximum(interval["Revenue_p95"], interval["Revenue_p90"])
    interval["COGS_p10"] = np.minimum(interval["COGS_p10"], interval["COGS"])
    interval["COGS_p90"] = np.maximum(interval["COGS_p90"], interval["COGS"])
    interval["COGS_p05"] = np.minimum(interval["COGS_p05"], interval["COGS_p10"])
    interval["COGS_p95"] = np.maximum(interval["COGS_p95"], interval["COGS_p90"])

    submission, interval, weekly_pred = apply_controlled_recovery_total(
        submission,
        interval,
        weekly_pred,
        requested_controlled_recovery_total(),
    )
    submission, interval, weekly_pred = apply_sample_submission_anchor(
        submission,
        interval,
        weekly_pred,
        sample,
    )

    out = submission[["date", "Revenue", "COGS"]].rename(columns={"date": "Date"})
    out["Revenue"] = out["Revenue"].clip(lower=0).round(2)
    out["COGS"] = out["COGS"].clip(lower=0).round(2)
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    interval_out = interval.rename(columns={"date": "Date"})
    for col in [c for c in interval_out.columns if c != "Date"]:
        interval_out[col] = interval_out[col].clip(lower=0).round(2)
    interval_out["Date"] = pd.to_datetime(interval_out["Date"]).dt.strftime("%Y-%m-%d")
    return out, weekly_pred, interval_out


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    if len(submission) != len(sample):
        raise ValueError(f"Submission row count mismatch: {len(submission)} vs {len(sample)}")
    if list(submission.columns) != ["Date", "Revenue", "COGS"]:
        raise ValueError(f"Submission columns are invalid: {list(submission.columns)}")
    sub_dates = pd.to_datetime(submission["Date"])
    if not sub_dates.equals(sample["Date"]):
        raise ValueError("Submission dates do not match sample_submission.csv")
    for col in ["Revenue", "COGS"]:
        if not np.isfinite(submission[col]).all():
            raise ValueError(f"Submission contains non-finite values in {col}")
        if (submission[col] < 0).any():
            raise ValueError(f"Submission contains negative values in {col}")


def coherence_summary(submission: pd.DataFrame, weekly_pred: pd.DataFrame) -> pd.DataFrame:
    daily = submission.copy()
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = add_calendar_columns(daily.rename(columns={"Date": "date"}), "date")
    counts = daily.groupby("week_id")["date"].transform("count")
    daily = daily[counts.eq(7)].copy()
    summary = daily.groupby("week_id", as_index=False).agg(
        Revenue=("Revenue", "sum"),
        COGS=("COGS", "sum"),
    )
    keep_cols = [
        "week_id",
        "revenue_w_base",
        "cogs_w_base",
        "revenue_w_pred",
        "cogs_w_pred",
        "revenue_lv3_multiplier_avg",
        "cogs_lv3_multiplier_avg",
    ]
    summary = summary.merge(weekly_pred[[c for c in keep_cols if c in weekly_pred.columns]], on="week_id", how="inner")
    summary["revenue_bottomup_drift"] = np.abs(summary["Revenue"] - summary["revenue_w_pred"]) / np.maximum(summary["revenue_w_pred"], EPS)
    summary["cogs_bottomup_drift"] = np.abs(summary["COGS"] - summary["cogs_w_pred"]) / np.maximum(summary["cogs_w_pred"], EPS)
    summary["revenue_weekly_sum_ratio"] = _safe_div_series(summary["Revenue"], summary["revenue_w_base"]).replace([np.inf, -np.inf], np.nan)
    summary["cogs_weekly_sum_ratio"] = _safe_div_series(summary["COGS"], summary["cogs_w_base"]).replace([np.inf, -np.inf], np.nan)
    summary["revenue_weekly_drift"] = (summary["revenue_weekly_sum_ratio"] - 1.0).abs()
    summary["cogs_weekly_drift"] = (summary["cogs_weekly_sum_ratio"] - 1.0).abs()
    summary["revenue_lv3_uplift"] = _safe_div_series(summary["revenue_w_pred"], summary["revenue_w_base"]) - 1.0
    summary["cogs_lv3_uplift"] = _safe_div_series(summary["cogs_w_pred"], summary["cogs_w_base"]) - 1.0
    return summary
