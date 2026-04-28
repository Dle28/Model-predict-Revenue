from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from .common import FOLDS
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


def _add_daily_slices(row: Dict[str, float | str], daily_eval: pd.DataFrame) -> None:
    if daily_eval.empty:
        return
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
    if high.any():
        row["daily_revenue_wape_high_bucket"] = _safe_wape(daily_eval.loc[high, "revenue"], daily_eval.loc[high, "Revenue"])
    if {"Revenue_p10", "Revenue_p90"}.issubset(daily_eval.columns):
        row["daily_revenue_p10_p90_coverage"] = float(
            daily_eval["revenue"].between(daily_eval["Revenue_p10"], daily_eval["Revenue_p90"]).mean()
        )
    if {"COGS_p10", "COGS_p90"}.issubset(daily_eval.columns):
        row["daily_cogs_p10_p90_coverage"] = float(
            daily_eval["cogs"].between(daily_eval["COGS_p10"], daily_eval["COGS_p90"]).mean()
        )


def run_backtests(daily: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tr_start, tr_end, va_start, va_end in FOLDS:
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

        for target, actual_col in [("revenue_w", "revenue_w"), ("cogs_w", "cogs_w")]:
            base_col = f"{target}_base"
            if base_col in pred.columns:
                row[f"{target}_avg_lv3_weekly_uplift"] = float((pred[f"{target}_pred"] / pred[base_col].clip(lower=1e-9) - 1.0).mean())

        val_daily_all = daily[daily["week_id"].isin(val_week_ids) & daily["revenue"].notna()].copy()
        daily_eval = val_daily_all[["date", "revenue", "cogs"]].merge(
            daily_pred[
                [
                    "date",
                    "Revenue",
                    "COGS",
                    "Revenue_lv3_multiplier",
                    "COGS_lv3_multiplier",
                    "Revenue_p10",
                    "Revenue_p90",
                    "COGS_p10",
                    "COGS_p90",
                ]
            ],
            on="date",
            how="inner",
        )
        row.update(metric_frame(daily_eval["revenue"], daily_eval["Revenue"], "daily_revenue"))
        row.update(metric_frame(daily_eval["cogs"], daily_eval["COGS"], "daily_cogs"))
        row["revenue_lv3_multiplier_max"] = float(daily_eval["Revenue_lv3_multiplier"].max())
        row["cogs_lv3_multiplier_max"] = float(daily_eval["COGS_lv3_multiplier"].max())
        _add_daily_slices(row, daily_eval)
        rows.append(row)
    return pd.DataFrame(rows)
