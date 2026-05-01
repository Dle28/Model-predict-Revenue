from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


EPS = 1e-9


def week_start_from_week_id(week_id: str) -> pd.Timestamp:
    year_str, week_str = str(week_id).split("-W")
    year = int(year_str)
    week = int(week_str)
    return pd.Timestamp(dt.date.fromisocalendar(year, week, 1))


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
FINAL_TRAIN_END_WEEK_ID = "2022-W51"

PRE_COVID_END_WEEK_ID = "2019-W52"
COVID_DROP_START_WEEK_ID = "2020-W01"
COVID_DROP_END_WEEK_ID = "2021-W52"
RECOVERY_START_WEEK_ID = "2022-W01"
RECOVERY_END_WEEK_ID = "2099-W52"
NORMALIZATION_START_WEEK_ID = "2024-W10"
COVID_START_DATE = week_start_from_week_id(COVID_DROP_START_WEEK_ID)
RECOVERY_START_DATE = week_start_from_week_id(RECOVERY_START_WEEK_ID)
RECOVERY_PROGRESS_TAU_WEEKS = 32.0


FOLDS = [
    # Diagnostic holdout: learn the post-2018 operating regime, then test 2022.
    # The final submission forecast still trains through FINAL_TRAIN_END_WEEK_ID.
    ("2019-W01", "2021-W52", "2022-W01", "2022-W51"),
]

YEARLY_ROLLING_ORIGIN_FOLDS = [
    ("2012-W28", "2018-W52", "2019-W01", "2019-W52"),
    ("2012-W28", "2019-W52", "2020-W01", "2020-W53"),
    ("2012-W28", "2020-W53", "2021-W01", "2021-W52"),
    ("2012-W28", "2021-W52", "2022-W01", "2022-W51"),
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
    "products.csv",
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
    recovery_progress = float(np.clip(1.0 - np.exp(-weeks_since_recovery_start / RECOVERY_PROGRESS_TAU_WEEKS), 0.0, 1.0))
    return {
        "pre_covid": float(week_id <= PRE_COVID_END_WEEK_ID),
        "covid_drop": float(COVID_DROP_START_WEEK_ID <= week_id <= COVID_DROP_END_WEEK_ID),
        "recovery_phase": float(RECOVERY_START_WEEK_ID <= week_id < NORMALIZATION_START_WEEK_ID),
        "normalization_phase": float(week_id >= NORMALIZATION_START_WEEK_ID),
        "weeks_since_covid_start": weeks_since_covid_start,
        "weeks_since_recovery_start": weeks_since_recovery_start,
        "recovery_progress": recovery_progress,
    }


def covid_regime_flag_frame(week_id: pd.Series, week_start: pd.Series) -> pd.DataFrame:
    week_id_series = pd.Series(week_id).astype(str)
    week_start_values = pd.to_datetime(week_start)
    if isinstance(week_start_values, pd.Series):
        week_start_series = week_start_values.reindex(week_id_series.index)
    else:
        week_start_series = pd.Series(week_start_values, index=week_id_series.index)

    weeks_since_covid_start = ((week_start_series - COVID_START_DATE).dt.days.astype(float) / 7.0).clip(lower=0.0)
    weeks_since_recovery_start = ((week_start_series - RECOVERY_START_DATE).dt.days.astype(float) / 7.0).clip(lower=0.0)
    recovery_progress = (1.0 - np.exp(-weeks_since_recovery_start / RECOVERY_PROGRESS_TAU_WEEKS)).clip(lower=0.0, upper=1.0)
    return pd.DataFrame(
        {
            "pre_covid": week_id_series.le(PRE_COVID_END_WEEK_ID).astype(float),
            "covid_drop": week_id_series.between(COVID_DROP_START_WEEK_ID, COVID_DROP_END_WEEK_ID).astype(float),
            "recovery_phase": (
                week_id_series.ge(RECOVERY_START_WEEK_ID) & week_id_series.lt(NORMALIZATION_START_WEEK_ID)
            ).astype(float),
            "normalization_phase": week_id_series.ge(NORMALIZATION_START_WEEK_ID).astype(float),
            "weeks_since_covid_start": weeks_since_covid_start.astype(float),
            "weeks_since_recovery_start": weeks_since_recovery_start.astype(float),
            "recovery_progress": recovery_progress.astype(float),
        },
        index=week_id_series.index,
    )


def covid_allocation_regime(week_id: pd.Series) -> pd.Series:
    labels = pd.Series("pre_covid", index=week_id.index, dtype=object)
    week_id = week_id.astype(str)
    labels = labels.mask(week_id.between(COVID_DROP_START_WEEK_ID, COVID_DROP_END_WEEK_ID), "covid_drop")
    labels = labels.mask(week_id.ge(RECOVERY_START_WEEK_ID) & week_id.lt(NORMALIZATION_START_WEEK_ID), "recovery_phase")
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


def _jd_from_date(day: int, month: int, year: int) -> int:
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    return day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045


def _date_from_jd(jd: int) -> tuple[int, int, int]:
    a = jd + 32044
    b = (4 * a + 3) // 146097
    c = a - (b * 146097) // 4
    d = (4 * c + 3) // 1461
    e = c - (1461 * d) // 4
    m = (5 * e + 2) // 153
    day = e - (153 * m + 2) // 5 + 1
    month = m + 3 - 12 * (m // 10)
    year = b * 100 + d - 4800 + m // 10
    return day, month, year


def _new_moon_day(k: int, time_zone: float = 7.0) -> int:
    t = k / 1236.85
    t2 = t * t
    t3 = t2 * t
    dr = math.pi / 180.0
    jd1 = 2415020.75933 + 29.53058868 * k + 0.0001178 * t2 - 0.000000155 * t3
    jd1 += 0.00033 * math.sin((166.56 + 132.87 * t - 0.009173 * t2) * dr)
    m = 359.2242 + 29.10535608 * k - 0.0000333 * t2 - 0.00000347 * t3
    mpr = 306.0253 + 385.81691806 * k + 0.0107306 * t2 + 0.00001236 * t3
    f = 21.2964 + 390.67050646 * k - 0.0016528 * t2 - 0.00000239 * t3
    c1 = (0.1734 - 0.000393 * t) * math.sin(m * dr) + 0.0021 * math.sin(2 * dr * m)
    c1 -= 0.4068 * math.sin(mpr * dr) + 0.0161 * math.sin(2 * dr * mpr)
    c1 -= 0.0004 * math.sin(3 * dr * mpr)
    c1 += 0.0104 * math.sin(2 * dr * f) - 0.0051 * math.sin((m + mpr) * dr)
    c1 -= 0.0074 * math.sin((m - mpr) * dr) + 0.0004 * math.sin((2 * f + m) * dr)
    c1 -= 0.0004 * math.sin((2 * f - m) * dr) - 0.0006 * math.sin((2 * f + mpr) * dr)
    c1 += 0.0010 * math.sin((2 * f - mpr) * dr) + 0.0005 * math.sin((2 * mpr + m) * dr)
    if t < -11:
        delta_t = 0.001 + 0.000839 * t + 0.0002261 * t2 - 0.00000845 * t3 - 0.000000081 * t * t3
    else:
        delta_t = -0.000278 + 0.000265 * t + 0.000262 * t2
    return int(jd1 + c1 - delta_t + 0.5 + time_zone / 24.0)


def _sun_longitude_segment(jdn: int, time_zone: float = 7.0) -> int:
    t = (jdn - 2451545.5 - time_zone / 24.0) / 36525.0
    t2 = t * t
    dr = math.pi / 180.0
    m = 357.52910 + 35999.05030 * t - 0.0001559 * t2 - 0.00000048 * t * t2
    l0 = 280.46645 + 36000.76983 * t + 0.0003032 * t2
    dl = (1.914600 - 0.004817 * t - 0.000014 * t2) * math.sin(dr * m)
    dl += (0.019993 - 0.000101 * t) * math.sin(2 * dr * m) + 0.000290 * math.sin(3 * dr * m)
    longitude = (l0 + dl) * dr
    longitude -= math.pi * 2 * math.floor(longitude / (math.pi * 2))
    return int(longitude / math.pi * 6)


def _lunar_month_11(year: int, time_zone: float = 7.0) -> int:
    off = _jd_from_date(31, 12, year) - 2415021
    k = int(off / 29.530588853)
    nm = _new_moon_day(k, time_zone)
    if _sun_longitude_segment(nm, time_zone) >= 9:
        nm = _new_moon_day(k - 1, time_zone)
    return nm


def _leap_month_offset(a11: int, time_zone: float = 7.0) -> int:
    k = int(0.5 + (a11 - 2415021.076998695) / 29.530588853)
    last = 0
    i = 1
    arc = _sun_longitude_segment(_new_moon_day(k + i, time_zone), time_zone)
    while arc != last and i < 14:
        last = arc
        i += 1
        arc = _sun_longitude_segment(_new_moon_day(k + i, time_zone), time_zone)
    return i - 1


def _vietnam_lunar_to_solar_date(day: int, month: int, year: int, leap: int = 0) -> pd.Timestamp:
    if month < 11:
        a11 = _lunar_month_11(year - 1)
        b11 = _lunar_month_11(year)
    else:
        a11 = _lunar_month_11(year)
        b11 = _lunar_month_11(year + 1)
    k = int(0.5 + (a11 - 2415021.076998695) / 29.530588853)
    off = month - 11
    if off < 0:
        off += 12
    if b11 - a11 > 365:
        leap_off = _leap_month_offset(a11)
        leap_month = leap_off - 2
        if leap_month < 0:
            leap_month += 12
        if leap and month != leap_month:
            raise ValueError(f"Invalid leap lunar month {month} for year {year}")
        if leap or off >= leap_off:
            off += 1
    month_start = _new_moon_day(k + off)
    solar_day, solar_month, solar_year = _date_from_jd(month_start + day - 1)
    return pd.Timestamp(dt.date(solar_year, solar_month, solar_day))


def _tet_window_dates(years: range) -> set[pd.Timestamp]:
    dates = set()
    for year in years:
        dt = _vietnam_lunar_to_solar_date(1, 1, int(year))
        for offset in range(-3, 4):
            dates.add(pd.Timestamp(dt + pd.Timedelta(days=offset)).normalize())
    return dates


def _hung_kings_dates(years: range) -> set[pd.Timestamp]:
    return {
        _vietnam_lunar_to_solar_date(10, 3, int(year)).normalize()
        for year in years
    }


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
    tet_dates = _tet_window_dates(years)
    hung_kings_dates = _hung_kings_dates(years)
    black_friday_dates = _black_friday_dates(years)
    lunar_holiday_dates = tet_dates | hung_kings_dates
    holiday_dates = lunar_holiday_dates | black_friday_dates
    out["is_holiday"] = (fixed_retail_holiday | out[date_col].dt.normalize().isin(holiday_dates)).astype(int)
    out["is_tet_window"] = normalized_date.isin(tet_dates).astype(int)
    out["is_hung_kings_day"] = normalized_date.isin(hung_kings_dates).astype(int)
    out["is_lunar_holiday"] = normalized_date.isin(lunar_holiday_dates).astype(int)
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
