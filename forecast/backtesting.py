from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .common import FOLDS, YEARLY_ROLLING_ORIGIN_FOLDS
from .final import forecast_daily_base_spikes
from .lv1 import metric_frame


WALK_FORWARD_CHUNK_WEEKS = 13


def _chunked(values: List[str], chunk_size: int) -> List[List[str]]:
    return [values[i : i + chunk_size] for i in range(0, len(values), chunk_size)]


def _previous_week_id(weekly: pd.DataFrame, week_id: str) -> str:
    ordered = weekly[["week_id", "week_start"]].drop_duplicates().sort_values("week_start").reset_index(drop=True)
    matches = ordered.index[ordered["week_id"].eq(week_id)]
    if len(matches) == 0 or int(matches[0]) == 0:
        return week_id
    return str(ordered.loc[int(matches[0]) - 1, "week_id"])


def _forecast_walk_forward_fold(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    train_start_week_id: str,
    initial_train_end_week_id: str,
    val_week_ids: List[str],
    chunk_weeks: int = WALK_FORWARD_CHUNK_WEEKS,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    daily_parts = []
    weekly_parts = []
    chunks = _chunked(val_week_ids, chunk_weeks)
    for chunk in chunks:
        train_end = initial_train_end_week_id if not weekly_parts else _previous_week_id(weekly, chunk[0])
        val_daily = daily[daily["week_id"].isin(chunk) & daily["revenue"].notna()].copy()
        if val_daily.empty:
            continue
        daily_pred, weekly_pred = forecast_daily_base_spikes(
            daily,
            weekly,
            val_daily,
            train_end,
            train_start_week_id,
        )
        daily_parts.append(daily_pred)
        weekly_parts.append(weekly_pred)
    if not daily_parts or not weekly_parts:
        return pd.DataFrame(), pd.DataFrame(), 0
    return pd.concat(daily_parts, ignore_index=True), pd.concat(weekly_parts, ignore_index=True), len(chunks)


def _safe_wape(actual: pd.Series, pred: pd.Series) -> float:
    metrics = metric_frame(actual.to_numpy(), pred.to_numpy(), "tmp")
    return float(metrics["tmp_wape"])


def _safe_bias(actual: pd.Series, pred: pd.Series) -> float:
    frame = pd.DataFrame({"actual": actual, "pred": pred}).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return float("nan")
    actual_sum = float(frame["actual"].sum())
    pred_sum = float(frame["pred"].sum())
    return float((pred_sum - actual_sum) / max(abs(actual_sum), 1e-9))


def _add_weekly_drift_metrics(row: Dict[str, float | str], pred: pd.DataFrame) -> None:
    for prefix in ["revenue_w", "cogs_w"]:
        pred_col = f"{prefix}_pred"
        base_col = f"{prefix}_base"
        if pred_col not in pred.columns or base_col not in pred.columns:
            continue
        ratio = (pred[pred_col] / pred[base_col].clip(lower=1e-9)).replace([np.inf, -np.inf], np.nan).dropna()
        if ratio.empty:
            continue
        drift = (ratio - 1.0).abs()
        row[f"{prefix}_weekly_sum_ratio_median"] = float(ratio.median())
        row[f"{prefix}_weekly_drift_mean"] = float(drift.mean())
        row[f"{prefix}_weekly_drift_p90"] = float(drift.quantile(0.90))
        row[f"{prefix}_weekly_drift_p95"] = float(drift.quantile(0.95))
        row[f"{prefix}_weekly_drift_max"] = float(drift.max())


EVENT_FLOOR_FLAGS = [
    ("is_holiday", "is_holiday"),
    ("is_tet_window", "is_tet_window"),
    ("is_black_friday_window", "is_black_friday_window"),
    ("is_double_day_sale", "is_double_day_sale"),
    ("is_1111_1212", "is_1111_1212"),
    ("is_year_end_window", "is_year_end_window"),
    ("is_new_year_window", "is_new_year_window"),
    ("has_promo", "active_promo_count"),
    ("strong_discount_promo", "avg_promo_discount_value"),
]


def _event_flag_mask(daily_eval: pd.DataFrame, flag_name: str, source_col: str) -> pd.Series:
    if source_col not in daily_eval.columns:
        return pd.Series(False, index=daily_eval.index)
    values = daily_eval[source_col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if flag_name == "strong_discount_promo":
        active = daily_eval.get("active_promo_count", pd.Series(0.0, index=daily_eval.index)).fillna(0.0).gt(0)
        return active & values.ge(15.0)
    return values.gt(0)


def _event_floor_validation_rows(fold: str, daily_eval: pd.DataFrame) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    if daily_eval.empty:
        return rows
    for target, actual_col in [("Revenue", "revenue"), ("COGS", "cogs")]:
        base_col = f"{target}_lv2_base"
        before_col = f"{target}_before_floor"
        after_col = f"{target}_after_floor"
        raw_mult_col = f"{target}_lv3_raw_multiplier"
        pred_mult_col = f"{target}_lv3_multiplier"
        if not {actual_col, base_col, before_col, after_col, raw_mult_col, pred_mult_col}.issubset(daily_eval.columns):
            continue
        for flag_name, source_col in EVENT_FLOOR_FLAGS:
            mask = _event_flag_mask(daily_eval, flag_name, source_col)
            if not mask.any():
                continue
            base = daily_eval.loc[mask, base_col].clip(lower=1e-9)
            actual_multiplier = daily_eval.loc[mask, actual_col] / base
            rows.append(
                {
                    "fold": fold,
                    "target": target,
                    "event_flag": flag_name,
                    "count": int(mask.sum()),
                    "floor_applied_rate": float(daily_eval.loc[mask, f"{target}_event_floor_applied"].mean())
                    if f"{target}_event_floor_applied" in daily_eval.columns
                    else np.nan,
                    "actual_multiplier_median": float(actual_multiplier.replace([np.inf, -np.inf], np.nan).median()),
                    "raw_pred_multiplier_median": float(daily_eval.loc[mask, raw_mult_col].median()),
                    "pred_multiplier_median": float(daily_eval.loc[mask, pred_mult_col].median()),
                    "WAPE_before_floor": _safe_wape(daily_eval.loc[mask, actual_col], daily_eval.loc[mask, before_col]),
                    "WAPE_after_floor": _safe_wape(daily_eval.loc[mask, actual_col], daily_eval.loc[mask, after_col]),
                }
            )
    return rows


def _add_daily_slices(row: Dict[str, float | str], daily_eval: pd.DataFrame) -> None:
    if daily_eval.empty:
        return
    row["daily_revenue_bias_all"] = _safe_bias(daily_eval["revenue"], daily_eval["Revenue"])
    row["daily_cogs_bias_all"] = _safe_bias(daily_eval["cogs"], daily_eval["COGS"])
    for weekday in range(1, 8):
        mask = pd.to_datetime(daily_eval["date"]).dt.isocalendar().day.astype(int).eq(weekday)
        if mask.any():
            row[f"daily_revenue_wape_wday_{weekday}"] = _safe_wape(daily_eval.loc[mask, "revenue"], daily_eval.loc[mask, "Revenue"])
    month = pd.to_datetime(daily_eval["date"]).dt.month.astype(int)
    for month_id in sorted(month.unique()):
        mask = month.eq(month_id)
        if mask.any():
            row[f"daily_revenue_wape_month_{month_id:02d}"] = _safe_wape(daily_eval.loc[mask, "revenue"], daily_eval.loc[mask, "Revenue"])
    q20 = daily_eval["revenue"].quantile(0.20)
    q80 = daily_eval["revenue"].quantile(0.80)
    low = daily_eval["revenue"].le(q20)
    high = daily_eval["revenue"].ge(q80)
    if low.any():
        row["daily_revenue_wape_low_bucket"] = _safe_wape(daily_eval.loc[low, "revenue"], daily_eval.loc[low, "Revenue"])
        row["daily_revenue_bias_low_bucket"] = _safe_bias(daily_eval.loc[low, "revenue"], daily_eval.loc[low, "Revenue"])
        row["daily_cogs_bias_low_bucket"] = _safe_bias(daily_eval.loc[low, "cogs"], daily_eval.loc[low, "COGS"])
    if high.any():
        row["daily_revenue_wape_high_bucket"] = _safe_wape(daily_eval.loc[high, "revenue"], daily_eval.loc[high, "Revenue"])
        row["daily_revenue_bias_high_bucket"] = _safe_bias(daily_eval.loc[high, "revenue"], daily_eval.loc[high, "Revenue"])
        row["daily_cogs_bias_high_bucket"] = _safe_bias(daily_eval.loc[high, "cogs"], daily_eval.loc[high, "COGS"])
    if {"Revenue_p10", "Revenue_p90"}.issubset(daily_eval.columns):
        row["daily_revenue_p10_p90_coverage"] = float(
            daily_eval["revenue"].between(daily_eval["Revenue_p10"], daily_eval["Revenue_p90"]).mean()
        )
    if {"COGS_p10", "COGS_p90"}.issubset(daily_eval.columns):
        row["daily_cogs_p10_p90_coverage"] = float(
            daily_eval["cogs"].between(daily_eval["COGS_p10"], daily_eval["COGS_p90"]).mean()
        )


def run_backtests(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    include_event_floor_table: bool = False,
    folds: List[Tuple[str, str, str, str]] | None = None,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    event_rows: List[Dict[str, float | str]] = []
    active_folds = folds if folds is not None else FOLDS
    for tr_start, tr_end, va_start, va_end in active_folds:
        train_mask = weekly["week_id"].between(tr_start, tr_end) & weekly["complete_target_week"]
        val_mask = weekly["week_id"].between(va_start, va_end) & weekly["complete_target_week"]
        val_week_ids = weekly.loc[val_mask, "week_id"].tolist()
        if not val_week_ids:
            continue

        daily_pred, weekly_pred, n_chunks = _forecast_walk_forward_fold(
            daily,
            weekly,
            tr_start,
            tr_end,
            val_week_ids,
            WALK_FORWARD_CHUNK_WEEKS,
        )
        if daily_pred.empty or weekly_pred.empty:
            continue

        actual = weekly[["week_id", "revenue_w", "cogs_w"]]
        pred = weekly_pred.merge(actual, on="week_id", how="left")

        row: Dict[str, float | str] = {
            "fold": f"{va_start}_{va_end}",
            "train_start": tr_start,
            "train_end": tr_end,
            "train_weeks": int(train_mask.sum()),
            "test_weeks": int(val_mask.sum()),
            "walk_forward_chunk_weeks": WALK_FORWARD_CHUNK_WEEKS,
            "walk_forward_chunks": n_chunks,
        }
        row.update(metric_frame(pred["revenue_w"], pred["revenue_w_base"], "weekly_base_revenue"))
        row.update(metric_frame(pred["cogs_w"], pred["cogs_w_base"], "weekly_base_cogs"))
        row.update(metric_frame(pred["revenue_w"], pred["revenue_w_pred"], "weekly_revenue"))
        row.update(metric_frame(pred["cogs_w"], pred["cogs_w_pred"], "weekly_cogs"))
        _add_weekly_drift_metrics(row, pred)

        for target, actual_col in [("revenue_w", "revenue_w"), ("cogs_w", "cogs_w")]:
            base_col = f"{target}_base"
            if base_col in pred.columns:
                row[f"{target}_avg_lv3_weekly_uplift"] = float((pred[f"{target}_pred"] / pred[base_col].clip(lower=1e-9) - 1.0).mean())

        val_daily_all = daily[daily["week_id"].isin(val_week_ids) & daily["revenue"].notna()].copy()
        event_cols = [
            "date",
            "revenue",
            "cogs",
            "is_holiday",
            "is_tet_window",
            "is_black_friday_window",
            "is_double_day_sale",
            "is_1111_1212",
            "is_year_end_window",
            "is_new_year_window",
            "active_promo_count",
            "avg_promo_discount_value",
        ]
        pred_cols = [
            "date",
            "Revenue",
            "COGS",
            "Revenue_lv2_base",
            "COGS_lv2_base",
            "Revenue_lv3_multiplier",
            "COGS_lv3_multiplier",
            "Revenue_lv3_raw_multiplier",
            "COGS_lv3_raw_multiplier",
            "Revenue_event_floor_applied",
            "COGS_event_floor_applied",
            "Revenue_before_floor",
            "COGS_before_floor",
            "Revenue_after_floor",
            "COGS_after_floor",
            "Revenue_p10",
            "Revenue_p90",
            "COGS_p10",
            "COGS_p90",
        ]
        daily_eval = val_daily_all[[c for c in event_cols if c in val_daily_all.columns]].merge(
            daily_pred[[c for c in pred_cols if c in daily_pred.columns]],
            on="date",
            how="inner",
        )
        row.update(
            metric_frame(
                daily_eval["revenue"],
                daily_eval["Revenue"],
                "daily_revenue",
                daily_eval["Revenue_p10"] if "Revenue_p10" in daily_eval.columns else None,
                daily_eval["Revenue_p90"] if "Revenue_p90" in daily_eval.columns else None,
            )
        )
        row.update(
            metric_frame(
                daily_eval["cogs"],
                daily_eval["COGS"],
                "daily_cogs",
                daily_eval["COGS_p10"] if "COGS_p10" in daily_eval.columns else None,
                daily_eval["COGS_p90"] if "COGS_p90" in daily_eval.columns else None,
            )
        )
        row["revenue_lv3_multiplier_max"] = float(daily_eval["Revenue_lv3_multiplier"].max())
        row["cogs_lv3_multiplier_max"] = float(daily_eval["COGS_lv3_multiplier"].max())
        _add_daily_slices(row, daily_eval)
        event_rows.extend(_event_floor_validation_rows(f"{va_start}_{va_end}", daily_eval))
        rows.append(row)
    metrics = pd.DataFrame(rows)
    if include_event_floor_table:
        return metrics, pd.DataFrame(event_rows)
    return metrics


def run_yearly_rolling_origin_cv(daily: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    result = run_backtests(
        daily,
        weekly,
        include_event_floor_table=False,
        folds=YEARLY_ROLLING_ORIGIN_FOLDS,
    )
    if isinstance(result, tuple):
        return result[0]
    return result
