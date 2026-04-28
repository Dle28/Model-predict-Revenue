from __future__ import annotations

from typing import Dict

import pandas as pd

from .allocation import allocate_and_reconcile, allocation_columns, historical_weight_tables
from .common import FOLDS
from .models import forecast_cogs_from_revenue, forecast_weekly_recursive, metric_frame


def run_backtests(daily: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tr_start, tr_end, va_start, va_end in FOLDS:
        train_mask = weekly["week_id"].between(tr_start, tr_end) & weekly["complete_target_week"]
        val_mask = weekly["week_id"].between(va_start, va_end) & weekly["complete_target_week"]
        val_week_ids = weekly.loc[val_mask, "week_id"].tolist()
        if not val_week_ids:
            continue

        pred_rev = forecast_weekly_recursive(weekly, "revenue_w", tr_end, val_week_ids)
        pred_cogs = forecast_cogs_from_revenue(weekly, pred_rev, tr_end, val_week_ids)
        pred = pred_rev.merge(pred_cogs, on=["week_id", "week_start"], how="inner")
        actual = weekly[["week_id", "revenue_w", "cogs_w"]]
        pred = pred.merge(actual, on="week_id", how="left")

        row: Dict[str, float | str] = {
            "fold": f"{va_start}_{va_end}",
            "train_start": tr_start,
            "train_end": tr_end,
            "train_weeks": int(train_mask.sum()),
            "test_weeks": int(val_mask.sum()),
        }
        row.update(metric_frame(pred["revenue_w"], pred["revenue_w_pred"], "weekly_revenue"))
        row.update(metric_frame(pred["cogs_w"], pred["cogs_w_pred"], "weekly_cogs"))

        train_end_date = weekly.loc[weekly["week_id"].eq(tr_end), "week_start"].max() + pd.Timedelta(days=6)
        val_daily = daily[daily["week_id"].isin(val_week_ids) & daily["revenue"].notna()].copy()

        rev_month_weekday, rev_weekday, rev_adjustments = historical_weight_tables(daily, "revenue", train_end_date)
        cogs_month_weekday, cogs_weekday, cogs_adjustments = historical_weight_tables(daily, "cogs", train_end_date)
        daily_rev = allocate_and_reconcile(
            val_daily[allocation_columns(val_daily)],
            pred.rename(columns={"revenue_w_pred": "pred"}),
            "pred",
            "revenue_pred",
            rev_month_weekday,
            rev_weekday,
            rev_adjustments,
        )
        daily_cogs = allocate_and_reconcile(
            val_daily[allocation_columns(val_daily)],
            pred.rename(columns={"cogs_w_pred": "pred"}),
            "pred",
            "cogs_pred",
            cogs_month_weekday,
            cogs_weekday,
            cogs_adjustments,
        )
        daily_pred = val_daily[["date", "revenue", "cogs"]].merge(daily_rev[["date", "revenue_pred"]], on="date").merge(
            daily_cogs[["date", "cogs_pred"]], on="date"
        )
        row.update(metric_frame(daily_pred["revenue"], daily_pred["revenue_pred"], "daily_revenue"))
        row.update(metric_frame(daily_pred["cogs"], daily_pred["cogs_pred"], "daily_cogs"))
        rows.append(row)
    return pd.DataFrame(rows)
