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


FOLDS = [
    # COVID holdout split:
    # Train uses complete pre-COVID ISO weeks only. Test covers the noisy COVID/recovery period.
    # 2022-W52 is excluded because the actual sales history ends on 2022-12-31, so that ISO week is partial.
    ("2012-W28", "2019-W52", "2020-W01", "2022-W51"),
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


def week_id_from_parts(iso_year: pd.Series, iso_week: pd.Series) -> pd.Series:
    return iso_year.astype(int).astype(str) + "-W" + iso_week.astype(int).astype(str).str.zfill(2)


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
    out["is_month_start"] = out[date_col].dt.is_month_start.astype(int)
    out["is_month_end"] = out[date_col].dt.is_month_end.astype(int)
    out["is_payday_window"] = out["day_of_month"].isin([25, 26, 27, 28, 29, 30, 31, 1, 2]).astype(int)
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
