from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .lv1 import RidgeLogModel


DIRECT_BLEND_WEIGHT = 0.80
DIRECT_TARGET_LAGS = [1, 7, 14, 28]
DIRECT_TARGET_WINDOWS = [7, 28]
DIRECT_ACTIVITY_COLUMNS = [
    "sessions",
    "unique_visitors",
    "page_views",
    "source_entropy",
    "orders_count",
    "orders_per_session",
    "new_customers",
    "active_promo_count",
    "avg_promo_discount_value",
    "stackable_promo_count",
    "discount_rate",
    "promo_intensity",
    "aov",
    "stockout_rate",
    "fill_rate",
    "return_rate",
    "refund_rate",
    "review_count",
    "avg_rating",
    "low_rating_share",
]


@dataclass
class DirectDailyModel:
    model: RidgeLogModel
    fill_values: Dict[str, float]
    train_end_date: pd.Timestamp
    target_col: str
    history_values: Dict[pd.Timestamp, float]


def _numeric_feature(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return df[col].replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def direct_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    date = pd.to_datetime(df["date"])
    iso_weekday = _numeric_feature(df, "iso_weekday")
    month = _numeric_feature(df, "month", 1.0)
    day_of_month = _numeric_feature(df, "day_of_month")
    day_of_year = _numeric_feature(df, "day_of_year")

    out = pd.DataFrame(index=df.index)
    out["time_index_day"] = _numeric_feature(df, "time_index_day")
    for weekday in range(1, 8):
        out[f"wday_{weekday}"] = iso_weekday.eq(weekday).astype(float)
    for month_id in range(1, 13):
        out[f"month_{month_id}"] = month.eq(month_id).astype(float)
    out["dom_sin"] = np.sin(2 * np.pi * day_of_month / 31.0)
    out["dom_cos"] = np.cos(2 * np.pi * day_of_month / 31.0)
    out["doy_sin"] = np.sin(2 * np.pi * day_of_year / 366.0)
    out["doy_cos"] = np.cos(2 * np.pi * day_of_year / 366.0)
    for col in [
        "is_payday_window",
        "is_holiday",
        "is_tet_window",
        "is_black_friday_window",
        "is_double_day_sale",
        "is_1111_1212",
        "is_year_end_window",
        "is_new_year_window",
        "is_month_start",
        "is_month_end",
    ]:
        out[col] = _numeric_feature(df, col)
    out["week_of_month"] = np.ceil(date.dt.day.astype(float) / 7.0)

    for col in DIRECT_ACTIVITY_COLUMNS:
        raw = _numeric_feature(df, col)
        out[col] = raw
        out[f"log1p_{col}"] = np.log1p(raw.clip(lower=0.0))
    for lag in DIRECT_TARGET_LAGS:
        raw = _numeric_feature(df, f"target_lag_{lag}d")
        out[f"log1p_target_lag_{lag}d"] = np.log1p(raw.clip(lower=0.0))
    for window in DIRECT_TARGET_WINDOWS:
        raw = _numeric_feature(df, f"target_ma_{window}d")
        out[f"log1p_target_ma_{window}d"] = np.log1p(raw.clip(lower=0.0))
    lag1 = _numeric_feature(df, "target_lag_1d")
    lag7 = _numeric_feature(df, "target_lag_7d")
    out["target_growth_1d"] = np.clip((lag1 - lag7) / np.maximum(np.abs(lag7), 1e-9), -5.0, 5.0)
    return out


def _history_map(daily: pd.DataFrame, target_col: str, end_date: pd.Timestamp) -> Dict[pd.Timestamp, float]:
    hist = daily[(daily["date"] <= end_date) & daily[target_col].notna()][["date", target_col]].copy()
    return {pd.Timestamp(row.date).normalize(): float(getattr(row, target_col)) for row in hist.itertuples(index=False)}


def _target_history_features_for_dates(dates: pd.Series, values_by_date: Dict[pd.Timestamp, float]) -> pd.DataFrame:
    rows = []
    for dt in pd.to_datetime(dates):
        key = pd.Timestamp(dt).normalize()
        row: Dict[str, float] = {}
        for lag in DIRECT_TARGET_LAGS:
            row[f"target_lag_{lag}d"] = values_by_date.get(key - pd.Timedelta(days=lag), np.nan)
        for window in DIRECT_TARGET_WINDOWS:
            vals = [
                values_by_date.get(key - pd.Timedelta(days=offset), np.nan)
                for offset in range(1, window + 1)
            ]
            arr = np.asarray(vals, dtype=float)
            row[f"target_ma_{window}d"] = float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows, index=dates.index)


def _attach_target_history_features(df: pd.DataFrame, values_by_date: Dict[pd.Timestamp, float]) -> pd.DataFrame:
    hist_features = _target_history_features_for_dates(df["date"], values_by_date)
    out = df.copy()
    for col in hist_features.columns:
        out[col] = hist_features[col]
    return out


def _activity_fill_values(train: pd.DataFrame) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for col in DIRECT_ACTIVITY_COLUMNS:
        if col not in train.columns:
            continue
        series = train[col].replace([np.inf, -np.inf], np.nan)
        positive = series[series > 0]
        values[col] = float(positive.median()) if len(positive) else float(series.median(skipna=True) or 0.0)
    return values


def _fill_unknown_future_activity(df: pd.DataFrame, direct_model: DirectDailyModel) -> pd.DataFrame:
    out = df.copy()
    future_mask = pd.to_datetime(out["date"]).gt(direct_model.train_end_date)
    for col, fill_value in direct_model.fill_values.items():
        if col not in out.columns:
            continue
        missing_like = future_mask & out[col].replace([np.inf, -np.inf], np.nan).fillna(0.0).le(0.0)
        out.loc[missing_like, col] = fill_value
    return out


def fit_direct_daily_model(
    daily: pd.DataFrame,
    target_col: str,
    train_start_date: pd.Timestamp | None,
    train_end_date: pd.Timestamp,
) -> DirectDailyModel:
    train = daily[(daily["date"] <= train_end_date) & daily[target_col].notna()].copy()
    if train_start_date is not None:
        train = train[train["date"] >= train_start_date].copy()
    fill_values = _activity_fill_values(train)
    history_values = _history_map(daily, target_col, train_end_date)
    train = _attach_target_history_features(train, history_values)
    model = RidgeLogModel(alpha=300.0).fit(direct_daily_features(train), np.log1p(train[target_col]))
    return DirectDailyModel(
        model=model,
        fill_values=fill_values,
        train_end_date=pd.Timestamp(train_end_date),
        target_col=target_col,
        history_values=history_values,
    )


def predict_direct_daily(daily: pd.DataFrame, direct_model: DirectDailyModel) -> np.ndarray:
    frame = _fill_unknown_future_activity(daily, direct_model).sort_values("date").copy()
    values_by_date = dict(direct_model.history_values)
    preds: List[float] = []
    for _, row in frame.iterrows():
        one = pd.DataFrame([row])
        one = _attach_target_history_features(one, values_by_date)
        pred = float(np.expm1(direct_model.model.predict(direct_daily_features(one))[0]))
        pred = max(pred, 0.0)
        values_by_date[pd.Timestamp(row["date"]).normalize()] = pred
        preds.append(pred)
    pred_series = pd.Series(preds, index=frame.index).reindex(daily.index)
    return pred_series.to_numpy(dtype=float)


def blend_direct_daily_forecast(
    daily: pd.DataFrame,
    daily_pred: pd.DataFrame,
    train_start_date: pd.Timestamp | None,
    train_end_date: pd.Timestamp,
    blend_weight: float = DIRECT_BLEND_WEIGHT,
) -> pd.DataFrame:
    out = daily_pred.copy()
    if not len(out):
        return out
    future = daily[daily["date"].isin(out["date"])].copy()
    future = out[["date"]].merge(future, on="date", how="left")
    blend = float(np.clip(blend_weight, 0.0, 1.0))
    for source_col, output_col in [("revenue", "Revenue"), ("cogs", "COGS")]:
        direct_model = fit_direct_daily_model(daily, source_col, train_start_date, train_end_date)
        direct_pred = predict_direct_daily(future, direct_model)
        base = out[output_col].astype(float).to_numpy()
        direct_pred = np.clip(direct_pred, base * 0.45, base * 1.90)
        blended = (1.0 - blend) * base + blend * direct_pred
        ratio = np.divide(blended, np.maximum(base, 1e-9))
        out[output_col] = blended
        for suffix in ["p10", "p90", "p05", "p95"]:
            col = f"{output_col}_{suffix}"
            if col in out.columns:
                out[col] = out[col].astype(float).to_numpy() * ratio
    return out
