from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


EPS = 1e-9


class Config:
    data_dir: Path = Path("Data")
    artifact_dir: Path = Path("artifacts")
    output_dir: Path = Path("outputs")
    submission_name: str = "submission.csv"

    @property
    def sample_submission_path(self) -> Path:
        return self.data_dir / "sample_submission.csv"

    @property
    def final_submission_path(self) -> Path:
        return self.output_dir / self.submission_name

    @property
    def submission_plot_path(self) -> Path:
        return self.output_dir / "submission_plot.html"


TARGETS = {
    "Revenue": "revenue",
    "COGS": "cogs",
}

FINAL_TRAIN_START_WEEK_ID = "2012-W28"
FINAL_TRAIN_END_WEEK_ID = "2022-W52"

PRE_COVID_END_WEEK_ID = "2019-W52"
COVID_DROP_START_WEEK_ID = "2020-W01"
COVID_DROP_END_WEEK_ID = "2022-W52"
RECOVERY_START_WEEK_ID = "2023-W01"
RECOVERY_END_WEEK_ID = "2023-W52"
NORMALIZATION_START_WEEK_ID = "2024-W01"
COVID_START_DATE = pd.Timestamp("2020-01-01")
RECOVERY_START_DATE = pd.Timestamp("2023-01-01")


FOLDS = [
    # Diagnostic holdout only. The final model trains from FINAL_TRAIN_START_WEEK_ID
    # through FINAL_TRAIN_END_WEEK_ID before forecasting 2023-2024.
    ("2017-W01", "2019-W52", "2020-W01", "2022-W51"),
]


REQUIRED_FILES = [
    "sales.csv",
    "sample_submission.csv",
    "orders.csv",
    "order_items.csv",
    "web_traffic.csv",
    "inventory.csv",
    "shipments.csv",
    "returns.csv",
    "reviews.csv",
    "customers.csv",
    "promotions.csv",
]


def checkpoint(name: str) -> None:
    print(f"\n=== {name} ===")


def safe_div(a: pd.Series | np.ndarray | float, b: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return a / np.maximum(np.abs(b), EPS)


def covid_regime_flags(week_id: str, week_start: pd.Timestamp) -> dict[str, float]:
    week_id = str(week_id)
    week_start = pd.Timestamp(week_start)
    weeks_since_covid_start = max(float((week_start - COVID_START_DATE).days) / 7.0, 0.0)
    weeks_since_recovery_start = max(float((week_start - RECOVERY_START_DATE).days) / 7.0, 0.0)
    recovery_progress = float(np.clip(weeks_since_recovery_start / 104.0, 0.0, 1.0))
    return {
        "pre_covid": float(week_id <= PRE_COVID_END_WEEK_ID),
        "covid_drop": float(COVID_DROP_START_WEEK_ID <= week_id <= COVID_DROP_END_WEEK_ID),
        "recovery_phase": float(RECOVERY_START_WEEK_ID <= week_id <= RECOVERY_END_WEEK_ID),
        "normalization_phase": float(week_id >= NORMALIZATION_START_WEEK_ID),
        "weeks_since_covid_start": weeks_since_covid_start,
        "weeks_since_recovery_start": weeks_since_recovery_start,
        "recovery_progress": recovery_progress,
    }


def covid_allocation_regime(week_id: pd.Series) -> pd.Series:
    labels = pd.Series("pre_covid", index=week_id.index, dtype=object)
    week_id = week_id.astype(str)
    labels = labels.mask(week_id.between(COVID_DROP_START_WEEK_ID, COVID_DROP_END_WEEK_ID), "covid_drop")
    labels = labels.mask(week_id.between(RECOVERY_START_WEEK_ID, RECOVERY_END_WEEK_ID), "recovery_phase")
    labels = labels.mask(week_id.ge(NORMALIZATION_START_WEEK_ID), "normalization_phase")
    return labels


def week_id_from_parts(iso_year: pd.Series, iso_week: pd.Series) -> pd.Series:
    return iso_year.astype(int).astype(str) + "-W" + iso_week.astype(int).astype(str).str.zfill(2)


def _black_friday_dates(years: range) -> set[pd.Timestamp]:
    dates = set()
    for year in years:
        november = pd.date_range(f"{year}-11-01", f"{year}-11-30", freq="D")
        thanksgiving = november[november.weekday == 3][3]
        dates.add(pd.Timestamp(thanksgiving + pd.Timedelta(days=1)).normalize())
        dates.add(pd.Timestamp(thanksgiving + pd.Timedelta(days=4)).normalize())
    return dates


def _tet_window_dates() -> set[pd.Timestamp]:
    tet_new_year_days = [
        "2012-01-23",
        "2013-02-10",
        "2014-01-31",
        "2015-02-19",
        "2016-02-08",
        "2017-01-28",
        "2018-02-16",
        "2019-02-05",
        "2020-01-25",
        "2021-02-12",
        "2022-02-01",
        "2023-01-22",
        "2024-02-10",
        "2025-01-29",
        "2026-02-17",
    ]
    dates = set()
    for dt in pd.to_datetime(tet_new_year_days):
        for offset in range(-3, 4):
            dates.add(pd.Timestamp(dt + pd.Timedelta(days=offset)).normalize())
    return dates


def add_calendar_columns(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    iso = out[date_col].dt.isocalendar()
    out["iso_year"] = iso["year"].astype(int)
    out["iso_week"] = iso["week"].astype(int)
    out["iso_weekday"] = iso["day"].astype(int)
    out["week_id"] = week_id_from_parts(out["iso_year"], out["iso_week"])
    out["week_start"] = out[date_col] - pd.to_timedelta(out["iso_weekday"] - 1, unit="D")
    out["month"] = out[date_col].dt.month.astype(int)
    out["quarter"] = out[date_col].dt.quarter.astype(int)
    out["day_of_month"] = out[date_col].dt.day.astype(int)
    out["day_of_year"] = out[date_col].dt.dayofyear.astype(int)
    out["is_weekend"] = out["iso_weekday"].isin([6, 7]).astype(int)
    out["week_position_in_month"] = ((out["day_of_month"] - 1) // 7 + 1).astype(int)
    out["is_month_start"] = out[date_col].dt.is_month_start.astype(int)
    out["is_month_end"] = out[date_col].dt.is_month_end.astype(int)
    out["is_payday_window"] = out["day_of_month"].isin([25, 26, 27, 28, 29, 30, 31, 1, 2]).astype(int)
    fixed_retail_holiday = (
        ((out["month"] == 1) & out["day_of_month"].eq(1))
        | ((out["month"] == 2) & out["day_of_month"].eq(14))
        | ((out["month"] == 3) & out["day_of_month"].eq(8))
        | ((out["month"] == 4) & out["day_of_month"].eq(30))
        | ((out["month"] == 5) & out["day_of_month"].eq(1))
        | ((out["month"] == 9) & out["day_of_month"].eq(2))
        | ((out["month"] == 9) & out["day_of_month"].eq(9))
        | ((out["month"] == 10) & out["day_of_month"].isin([10, 20]))
        | ((out["month"] == 11) & out["day_of_month"].eq(11))
        | ((out["month"] == 12) & out["day_of_month"].isin([12, 24, 25, 31]))
    )
    years = range(int(out["iso_year"].min()) - 1, int(out["iso_year"].max()) + 2)
    normalized_date = out[date_col].dt.normalize()
    tet_dates = _tet_window_dates()
    black_friday_dates = _black_friday_dates(years)
    holiday_dates = tet_dates | black_friday_dates
    out["is_holiday"] = (fixed_retail_holiday | out[date_col].dt.normalize().isin(holiday_dates)).astype(int)
    out["is_tet_window"] = normalized_date.isin(tet_dates).astype(int)
    out["is_black_friday_window"] = normalized_date.isin(black_friday_dates).astype(int)
    out["is_double_day_sale"] = ((out["month"] == out["day_of_month"]) & out["month"].isin([9, 10, 11, 12])).astype(int)
    out["is_1111_1212"] = (
        ((out["month"] == 11) & out["day_of_month"].eq(11))
        | ((out["month"] == 12) & out["day_of_month"].eq(12))
    ).astype(int)
    out["is_year_end_window"] = ((out["month"] == 12) & out["day_of_month"].ge(15)).astype(int)
    out["is_new_year_window"] = ((out["month"] == 1) & out["day_of_month"].le(7)).astype(int)
    return out


def read_csv(path: Path, parse_dates: Optional[List[str]] = None, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=parse_dates, usecols=usecols, low_memory=False)


def validate_inputs(cfg: Config) -> pd.DataFrame:
    missing = [name for name in REQUIRED_FILES if not (cfg.data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required data files: {missing}")

    sample = read_csv(cfg.sample_submission_path, parse_dates=["Date"])
    expected_cols = ["Date", "Revenue", "COGS"]
    if list(sample.columns) != expected_cols:
        raise ValueError(f"sample_submission.csv columns must be {expected_cols}, got {list(sample.columns)}")
    if sample["Date"].duplicated().any():
        raise ValueError("sample_submission.csv contains duplicate Date values")
    if not sample["Date"].is_monotonic_increasing:
        raise ValueError("sample_submission.csv Date column must be sorted ascending")

    print(
        "sample_submission:",
        len(sample),
        "rows",
        sample["Date"].min().date(),
        "->",
        sample["Date"].max().date(),
    )
    return sample
