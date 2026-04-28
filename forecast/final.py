from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .allocation import allocate_and_reconcile, allocation_columns, daily_seasonal_shim, historical_weight_tables
from .common import EPS, add_calendar_columns
from .models import forecast_cogs_from_revenue, forecast_weekly_recursive


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
    forecast_week_ids = complete_future_daily["week_id"].drop_duplicates().tolist()

    pred_rev = forecast_weekly_recursive(weekly, "revenue_w", "2022-W52", forecast_week_ids)
    pred_cogs = forecast_cogs_from_revenue(weekly, pred_rev, "2022-W52", forecast_week_ids)
    weekly_pred = pred_rev.merge(pred_cogs, on=["week_id", "week_start"], how="inner")

    train_end_date = pd.Timestamp("2022-12-31")
    rev_month_weekday, rev_weekday, rev_adjustments = historical_weight_tables(daily, "revenue", train_end_date)
    cogs_month_weekday, cogs_weekday, cogs_adjustments = historical_weight_tables(daily, "cogs", train_end_date)

    topdown_rev = allocate_and_reconcile(
        complete_future_daily[allocation_columns(complete_future_daily)],
        weekly_pred.rename(columns={"revenue_w_pred": "pred"}),
        "pred",
        "Revenue",
        rev_month_weekday,
        rev_weekday,
        rev_adjustments,
    )
    topdown_cogs = allocate_and_reconcile(
        complete_future_daily[allocation_columns(complete_future_daily)],
        weekly_pred.rename(columns={"cogs_w_pred": "pred"}),
        "pred",
        "COGS",
        cogs_month_weekday,
        cogs_weekday,
        cogs_adjustments,
    )
    topdown = topdown_rev[["date", "Revenue"]].merge(topdown_cogs[["date", "COGS"]], on="date")

    submission = submission.drop(columns=["Revenue", "COGS"]).merge(
        submission[["date", "Revenue", "COGS"]], on="date", how="left"
    )
    submission = submission.set_index("date")
    topdown = topdown.set_index("date")
    submission.loc[topdown.index, "Revenue"] = topdown["Revenue"]
    submission.loc[topdown.index, "COGS"] = topdown["COGS"]
    submission = submission.reset_index()

    interval = submission[["date", "Revenue", "COGS"]].copy()
    interval["Revenue_p10"] = interval["Revenue"] * 0.85
    interval["Revenue_p90"] = interval["Revenue"] * 1.15
    interval["Revenue_p05"] = interval["Revenue"] * 0.78
    interval["Revenue_p95"] = interval["Revenue"] * 1.25
    interval["COGS_p10"] = interval["COGS"] * 0.85
    interval["COGS_p90"] = interval["COGS"] * 1.15
    interval["COGS_p05"] = interval["COGS"] * 0.78
    interval["COGS_p95"] = interval["COGS"] * 1.25

    interval_specs = [
        ("revenue_w_pred_p10", "Revenue_p10", rev_month_weekday, rev_weekday, rev_adjustments),
        ("revenue_w_pred_p90", "Revenue_p90", rev_month_weekday, rev_weekday, rev_adjustments),
        ("revenue_w_pred_p05", "Revenue_p05", rev_month_weekday, rev_weekday, rev_adjustments),
        ("revenue_w_pred_p95", "Revenue_p95", rev_month_weekday, rev_weekday, rev_adjustments),
        ("cogs_w_pred_p10", "COGS_p10", cogs_month_weekday, cogs_weekday, cogs_adjustments),
        ("cogs_w_pred_p90", "COGS_p90", cogs_month_weekday, cogs_weekday, cogs_adjustments),
        ("cogs_w_pred_p05", "COGS_p05", cogs_month_weekday, cogs_weekday, cogs_adjustments),
        ("cogs_w_pred_p95", "COGS_p95", cogs_month_weekday, cogs_weekday, cogs_adjustments),
    ]
    interval = interval.set_index("date")
    for pred_col, out_col, month_weekday, weekday, adjustments in interval_specs:
        if pred_col not in weekly_pred.columns:
            continue
        allocated = allocate_and_reconcile(
            complete_future_daily[allocation_columns(complete_future_daily)],
            weekly_pred.rename(columns={pred_col: "pred"}),
            "pred",
            out_col,
            month_weekday,
            weekday,
            adjustments,
        ).set_index("date")
        interval.loc[allocated.index, out_col] = allocated[out_col]
    interval = interval.reset_index()
    interval["Revenue_p10"] = np.minimum(interval["Revenue_p10"], interval["Revenue"])
    interval["Revenue_p90"] = np.maximum(interval["Revenue_p90"], interval["Revenue"])
    interval["Revenue_p05"] = np.minimum(interval["Revenue_p05"], interval["Revenue_p10"])
    interval["Revenue_p95"] = np.maximum(interval["Revenue_p95"], interval["Revenue_p90"])
    interval["COGS_p10"] = np.minimum(interval["COGS_p10"], interval["COGS"])
    interval["COGS_p90"] = np.maximum(interval["COGS_p90"], interval["COGS"])
    interval["COGS_p05"] = np.minimum(interval["COGS_p05"], interval["COGS_p10"])
    interval["COGS_p95"] = np.maximum(interval["COGS_p95"], interval["COGS_p90"])

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
    summary = summary.merge(weekly_pred[["week_id", "revenue_w_pred", "cogs_w_pred"]], on="week_id", how="inner")
    summary["revenue_drift"] = np.abs(summary["Revenue"] - summary["revenue_w_pred"]) / np.maximum(summary["revenue_w_pred"], EPS)
    summary["cogs_drift"] = np.abs(summary["COGS"] - summary["cogs_w_pred"]) / np.maximum(summary["cogs_w_pred"], EPS)
    return summary
