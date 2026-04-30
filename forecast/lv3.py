from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .common import EPS, covid_regime_flag_frame, safe_div
from .lv1 import RidgeLogModel


LV3_MULTIPLIER_MIN = 0.05
LV3_MULTIPLIER_MAX = 4.00
LV3_INTERVAL_SCALE = 1.40
LV3_INTERVAL_HORIZON_DAILY_SLOPE = 0.015
LV3_INTERVAL_VALIDATION_DAYS = 91
TARGET_HISTORY_LAGS = [7, 14, 28]
TARGET_HISTORY_WINDOWS = [7, 28]
LV3_STRONG_EVENT_COLUMNS = [
    "is_1111_1212",
    "is_tet_window",
    "is_black_friday_window",
    "is_double_day_sale",
]
LV3_EVENT_FLOOR_QUANTILE = 0.40
LV3_EVENT_FLOOR_MIN_SAMPLES = 2
LV3_EVENT_BASE_MODES = {"lv2", "weekly_avg"}
DEFAULT_LV3_EVENT_BASE_MODE = "lv2"
PLANNED_PROMO_COLUMNS = {
    "active_promo_count",
    "promo_flag",
    "avg_promo_discount_value",
    "stackable_promo_count",
}

SPIKE_ACTIVITY_COLUMNS = [
    "active_promo_count",
    "promo_flag",
    "avg_promo_discount_value",
    "stackable_promo_count",
    "revenue_lag_7d",
    "revenue_lag_14d",
    "revenue_lag_28d",
    "revenue_ma_7d",
    "revenue_ma_28d",
    "revenue_shock_7d",
    "cogs_lag_7d",
    "cogs_lag_14d",
    "cogs_lag_28d",
    "cogs_ma_7d",
    "cogs_ma_28d",
    "cogs_shock_7d",
]

FUTURE_TARGET_HISTORY_COLUMNS = {
    "revenue_lag_7d",
    "revenue_lag_14d",
    "revenue_lag_28d",
    "revenue_ma_7d",
    "revenue_ma_28d",
    "revenue_shock_7d",
    "cogs_lag_7d",
    "cogs_lag_14d",
    "cogs_lag_28d",
    "cogs_ma_7d",
    "cogs_ma_28d",
    "cogs_shock_7d",
}

LV3_MONOTONE_CONSTRAINTS = {
    "active_promo_count": 1,
    "log1p_active_promo_count": 1,
    "avg_promo_discount_value": 1,
    "log1p_avg_promo_discount_value": 1,
    "stackable_promo_count": 1,
    "log1p_stackable_promo_count": 1,
}
LV3_MODEL_BACKEND_ALIASES = {
    "lgb": "lightgbm",
    "lgbm": "lightgbm",
    "xgb": "xgboost",
}
LV3_MODEL_BACKENDS = {"ridge", "lightgbm", "xgboost", "auto"}
DEFAULT_LV3_MODEL_BACKEND = "xgboost"


def _numeric_feature(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return df[col].replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def lv3_monotone_constraints(features: list[str]) -> list[int]:
    return [LV3_MONOTONE_CONSTRAINTS.get(col, 0) for col in features]


def requested_lv3_model_backend() -> str:
    raw = os.environ.get("FORECAST_LV3_MODEL_BACKEND", os.environ.get("FORECAST_MODEL_BACKEND", DEFAULT_LV3_MODEL_BACKEND))
    backend = raw.strip().lower()
    backend = LV3_MODEL_BACKEND_ALIASES.get(backend, backend)
    return backend if backend in LV3_MODEL_BACKENDS else DEFAULT_LV3_MODEL_BACKEND


def requested_lv3_event_base_mode() -> str:
    mode = os.environ.get("FORECAST_LV3_EVENT_BASE_MODE", DEFAULT_LV3_EVENT_BASE_MODE).strip().lower()
    return mode if mode in LV3_EVENT_BASE_MODES else DEFAULT_LV3_EVENT_BASE_MODE


def requested_lv3_event_floor_enabled() -> bool:
    raw = os.environ.get("FORECAST_LV3_EVENT_FLOOR", "").strip().lower()
    return raw in {"on", "1", "true", "yes"}


def fit_lv3_xgboost_regressor(X: pd.DataFrame, y: pd.Series) -> tuple[Any, str]:
    import xgboost as xgb  # type: ignore

    constraints = tuple(lv3_monotone_constraints(list(X.columns)))
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        learning_rate=0.035,
        n_estimators=220,
        max_depth=3,
        min_child_weight=20,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.05,
        reg_lambda=4.0,
        monotone_constraints=constraints,
        random_state=42,
        n_jobs=0,
    )
    model.fit(X, y)
    return model, "xgboost_monotone"


def fit_lv3_lightgbm_regressor(X: pd.DataFrame, y: pd.Series) -> tuple[Any, str]:
    import lightgbm as lgb  # type: ignore

    model = lgb.LGBMRegressor(
        objective="regression",
        learning_rate=0.035,
        n_estimators=160,
        num_leaves=11,
        min_child_samples=45,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.05,
        reg_lambda=4.0,
        monotone_constraints=lv3_monotone_constraints(list(X.columns)),
        monotone_constraints_method="intermediate",
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)
    return model, "lightgbm_monotone"


def fit_lv3_log_regressor(X: pd.DataFrame, y: pd.Series, alpha: float = 100.0) -> tuple[Any, str]:
    if len(X) < 50:
        return RidgeLogModel(alpha=alpha).fit(X, y), "ridge"
    requested_backend = requested_lv3_model_backend()
    if requested_backend in {"xgboost", "auto"}:
        try:
            return fit_lv3_xgboost_regressor(X, y)
        except Exception:
            if requested_backend == "xgboost":
                return RidgeLogModel(alpha=alpha).fit(X, y), "ridge"
    try:
        if requested_backend in {"lightgbm", "auto"}:
            return fit_lv3_lightgbm_regressor(X, y)
    except Exception:
        pass
    return RidgeLogModel(alpha=alpha).fit(X, y), "ridge"


def spike_features(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        date = pd.to_datetime(df["date"])
    else:
        date = pd.Series(pd.NaT, index=df.index)
    iso_weekday = _numeric_feature(df, "iso_weekday")
    month = _numeric_feature(df, "month", 1.0)
    day_of_month = _numeric_feature(df, "day_of_month")
    if "day_of_month" not in df.columns and "date" in df.columns:
        day_of_month = date.dt.day.astype(float)
    day_of_year = _numeric_feature(df, "day_of_year")
    if "day_of_year" not in df.columns and "date" in df.columns:
        day_of_year = date.dt.dayofyear.astype(float)

    lv2_base = _numeric_feature(df, "lv2_base_value")
    multiplier_base = _numeric_feature(df, "lv3_multiplier_base_value")
    multiplier_base = multiplier_base.where(multiplier_base.gt(EPS), lv2_base)
    features: Dict[str, Any] = {
        "bias": 1.0,
        "log_lv2_base": np.log1p(lv2_base),
        "log_lv3_multiplier_base": np.log1p(multiplier_base),
    }
    if "week_id" in df.columns:
        week_start = pd.to_datetime(df["week_start"]) if "week_start" in df.columns else date
        flag_frame = covid_regime_flag_frame(df["week_id"].astype(str), week_start)
        for col in [
            "pre_covid",
            "covid_drop",
            "recovery_phase",
            "normalization_phase",
            "recovery_progress",
        ]:
            features[col] = flag_frame[col].astype(float)
    else:
        for col in [
            "pre_covid",
            "covid_drop",
            "recovery_phase",
            "normalization_phase",
            "recovery_progress",
        ]:
            features[col] = 0.0
    if "week_id" in df.columns:
        week_sum = lv2_base.groupby(df["week_id"]).transform("sum")
        week_mean = lv2_base.groupby(df["week_id"]).transform("mean")
        features["base_share_in_week"] = safe_div(lv2_base, week_sum).replace([np.inf, -np.inf], np.nan).fillna(1.0 / 7.0)
        features["base_vs_week_mean"] = safe_div(lv2_base, week_mean).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        features["base_rank_in_week"] = lv2_base.groupby(df["week_id"]).rank(pct=True).fillna(0.5)
    else:
        features["base_share_in_week"] = 1.0 / 7.0
        features["base_vs_week_mean"] = 1.0
        features["base_rank_in_week"] = 0.5
    weekly_base_value = _numeric_feature(df, "weekly_base_value")
    if "week_id" in df.columns:
        weekly_base_value = weekly_base_value.where(weekly_base_value.gt(EPS), lv2_base.groupby(df["week_id"]).transform("sum"))
    else:
        weekly_base_value = weekly_base_value.where(weekly_base_value.gt(EPS), lv2_base * 7.0)
    if "pre_covid_baseline_same_week" in df.columns:
        pre_covid_baseline = df["pre_covid_baseline_same_week"].replace([np.inf, -np.inf], np.nan).astype(float)
    else:
        pre_covid_baseline = pd.Series(np.nan, index=df.index, dtype=float)
    base_vs_precovid = safe_div(weekly_base_value, pre_covid_baseline).replace([np.inf, -np.inf], np.nan)
    base_vs_precovid_baseline = base_vs_precovid.fillna(1.0).clip(lower=0.0, upper=2.0)
    recovery_gap = (1.0 - base_vs_precovid_baseline).clip(lower=-0.5, upper=0.7)
    features["base_vs_precovid_baseline"] = base_vs_precovid_baseline
    features["recovery_gap"] = recovery_gap
    features["lv3_multiplier_base_ratio"] = safe_div(multiplier_base, lv2_base).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 5.0)
    features["lv3_uses_weekly_avg_event_base"] = _numeric_feature(df, "lv3_uses_weekly_avg_event_base")
    for weekday in range(1, 8):
        features[f"wday_{weekday}"] = iso_weekday.eq(weekday).astype(float)
    for month_id in range(1, 13):
        features[f"month_{month_id}"] = month.eq(month_id).astype(float)
    features["dom_sin"] = np.sin(2 * np.pi * day_of_month / 31.0)
    features["dom_cos"] = np.cos(2 * np.pi * day_of_month / 31.0)
    features["doy_sin"] = np.sin(2 * np.pi * day_of_year / 366.0)
    features["doy_cos"] = np.cos(2 * np.pi * day_of_year / 366.0)
    is_payday_window = _numeric_feature(df, "is_payday_window")
    is_holiday = _numeric_feature(df, "is_holiday")
    is_tet_window = _numeric_feature(df, "is_tet_window")
    is_black_friday_window = _numeric_feature(df, "is_black_friday_window")
    is_double_day_sale = (
        _numeric_feature(df, "is_double_day_sale")
        if "is_double_day_sale" in df.columns
        else ((month == day_of_month) & month.isin([9, 10, 11, 12])).astype(float)
    )
    is_1111_1212 = (
        _numeric_feature(df, "is_1111_1212")
        if "is_1111_1212" in df.columns
        else (((month == 11) & (day_of_month == 11)) | ((month == 12) & (day_of_month == 12))).astype(float)
    )
    is_year_end_window = (
        _numeric_feature(df, "is_year_end_window")
        if "is_year_end_window" in df.columns
        else ((month == 12) & day_of_month.ge(15)).astype(float)
    )
    is_new_year_window = (
        _numeric_feature(df, "is_new_year_window")
        if "is_new_year_window" in df.columns
        else ((month == 1) & day_of_month.le(7)).astype(float)
    )
    features.update(
        {
            "is_payday_window": is_payday_window,
            "is_holiday": is_holiday,
            "is_month_start": _numeric_feature(df, "is_month_start"),
            "is_month_end": _numeric_feature(df, "is_month_end"),
            "is_tet_window": is_tet_window,
            "is_black_friday_window": is_black_friday_window,
            "is_double_day_sale": is_double_day_sale,
            "is_1111_1212": is_1111_1212,
            "is_year_end_window": is_year_end_window,
            "is_new_year_window": is_new_year_window,
        }
    )

    for col in SPIKE_ACTIVITY_COLUMNS:
        raw = _numeric_feature(df, col)
        features[col] = raw
        features[f"log1p_{col}"] = np.log1p(raw.clip(lower=0.0))
    revenue_lag7 = _numeric_feature(df, "revenue_lag_7d")
    cogs_lag7 = _numeric_feature(df, "cogs_lag_7d")
    target_lag7 = revenue_lag7.where(revenue_lag7.gt(0), cogs_lag7)
    features["lag7_vs_base"] = safe_div(target_lag7, lv2_base).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 5.0)
    event_intensity = (
        is_holiday
        + is_tet_window
        + is_black_friday_window
        + is_double_day_sale
        + is_1111_1212
        + is_payday_window
        + _numeric_feature(df, "active_promo_count").clip(lower=0.0, upper=3.0)
    )
    features["event_intensity"] = event_intensity
    return pd.DataFrame(features, index=df.index).copy()


class SpikeMultiplierModel:
    def __init__(
        self,
        target_col: str,
        alpha: float = 100.0,
        min_multiplier: float = LV3_MULTIPLIER_MIN,
        max_multiplier: float = LV3_MULTIPLIER_MAX,
    ) -> None:
        self.target_col = target_col
        self.alpha = alpha
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self.model: Optional[Any] = None
        self.backend = "ridge"
        self.fallback_log_multiplier = 0.0
        self.abs_log_residual_q80 = 0.18
        self.abs_log_residual_q90 = 0.28
        self.train_end_date: Optional[pd.Timestamp] = None
        self.activity_fill_values: Dict[str, float] = {}
        self.history_values: Dict[pd.Timestamp, float] = {}
        self.event_floor_multipliers: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, multiplier: pd.Series) -> "SpikeMultiplierModel":
        y = np.log(multiplier.clip(lower=self.min_multiplier, upper=self.max_multiplier))
        finite = y.replace([np.inf, -np.inf], np.nan).notna()
        X = X.loc[finite].copy()
        y = y.loc[finite]
        if len(y):
            self.fallback_log_multiplier = float(np.median(y))
        if len(y) >= 20:
            self.model, self.backend = fit_lv3_log_regressor(X, y, alpha=self.alpha)
            pred = self.model.predict(X)
            residual = y.to_numpy() - pred
            residual = residual[np.isfinite(residual)]
            if len(residual):
                self.abs_log_residual_q80 = float(np.quantile(np.abs(residual), 0.80))
                self.abs_log_residual_q90 = float(np.quantile(np.abs(residual), 0.90))
        return self

    def set_interval_residuals(self, residual: pd.Series | np.ndarray) -> None:
        residual_values = np.asarray(residual, dtype=float)
        residual_values = residual_values[np.isfinite(residual_values)]
        if len(residual_values):
            self.abs_log_residual_q80 = float(np.quantile(np.abs(residual_values), 0.80))
            self.abs_log_residual_q90 = float(np.quantile(np.abs(residual_values), 0.90))

    def predict_multiplier(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            pred_log = np.full(len(X), self.fallback_log_multiplier, dtype=float)
        else:
            pred_log = np.asarray(self.model.predict(X), dtype=float)
        return np.exp(pred_log).clip(self.min_multiplier, self.max_multiplier)

    def interval_multiplier(
        self,
        X: pd.DataFrame,
        coverage: float = 0.80,
        dates: Optional[pd.Series] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            pred_log = np.full(len(X), self.fallback_log_multiplier, dtype=float)
        else:
            pred_log = np.asarray(self.model.predict(X), dtype=float)
        q = (self.abs_log_residual_q80 if coverage <= 0.80 else self.abs_log_residual_q90) * LV3_INTERVAL_SCALE
        if dates is not None and self.train_end_date is not None:
            date_values = pd.Series(pd.to_datetime(dates))
            days_ahead = (date_values - self.train_end_date).dt.days.clip(lower=0).to_numpy(dtype=float)
            q = q * (1.0 + LV3_INTERVAL_HORIZON_DAILY_SLOPE * days_ahead)
        lo = np.exp(pred_log - q).clip(self.min_multiplier, self.max_multiplier)
        hi = np.exp(pred_log + q).clip(self.min_multiplier, self.max_multiplier)
        return lo, hi


def calibrate_lv3_interval_from_holdout(
    model: SpikeMultiplierModel,
    train: pd.DataFrame,
    multiplier: pd.Series,
) -> None:
    if len(train) < 120:
        return
    dates = pd.to_datetime(train["date"])
    val_start = dates.max() - pd.Timedelta(days=LV3_INTERVAL_VALIDATION_DAYS - 1)
    y_log = np.log(multiplier.clip(lower=model.min_multiplier, upper=model.max_multiplier))
    finite = y_log.replace([np.inf, -np.inf], np.nan).notna()
    train_mask = dates.lt(val_start) & finite
    val_mask = dates.ge(val_start) & finite
    if train_mask.sum() < 50 or val_mask.sum() < 14:
        return
    try:
        holdout_model, _ = fit_lv3_log_regressor(
            spike_features(train.loc[train_mask]),
            y_log.loc[train_mask],
            alpha=model.alpha,
        )
        pred_log = np.asarray(holdout_model.predict(spike_features(train.loc[val_mask])), dtype=float)
        model.set_interval_residuals(y_log.loc[val_mask].to_numpy() - pred_log)
    except Exception:
        return


def _prepare_base_join(
    daily: pd.DataFrame,
    base_daily: pd.DataFrame,
    base_col: str,
) -> pd.DataFrame:
    join_cols = ["date", base_col]
    for col in ["weekly_base_value", "pre_covid_baseline_same_week", "recovery_progress"]:
        if col in base_daily.columns:
            join_cols.append(col)
    out = daily.merge(base_daily[join_cols], on="date", how="left")
    out = out.rename(columns={base_col: "lv2_base_value"})
    return out


def _strong_event_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for col in LV3_STRONG_EVENT_COLUMNS:
        if col in df.columns:
            mask = mask | _numeric_feature(df, col).gt(0.0)
    return mask


def _attach_multiplier_base(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lv2_base = _numeric_feature(out, "lv2_base_value")
    weekly_base = _numeric_feature(out, "weekly_base_value")
    weekly_avg_base = (weekly_base / 7.0).where(weekly_base.gt(EPS), lv2_base)
    strong_event = (
        _strong_event_mask(out)
        if requested_lv3_event_base_mode() == "weekly_avg"
        else pd.Series(False, index=out.index)
    )
    multiplier_base = lv2_base.where(~strong_event, weekly_avg_base)
    out["lv3_multiplier_base_value"] = multiplier_base.clip(lower=EPS)
    out["lv3_uses_weekly_avg_event_base"] = strong_event.astype(float)
    return out


def _event_floor_multipliers(train: pd.DataFrame, multiplier: pd.Series) -> Dict[str, float]:
    if not requested_lv3_event_floor_enabled():
        return {}
    floors: Dict[str, float] = {}
    clean_multiplier = multiplier.replace([np.inf, -np.inf], np.nan).astype(float)
    for col in LV3_STRONG_EVENT_COLUMNS:
        if col not in train.columns:
            continue
        mask = _numeric_feature(train, col).gt(0.0) & clean_multiplier.notna()
        if int(mask.sum()) < LV3_EVENT_FLOOR_MIN_SAMPLES:
            continue
        floor = float(clean_multiplier.loc[mask].quantile(LV3_EVENT_FLOOR_QUANTILE))
        if np.isfinite(floor) and floor > EPS:
            floors[col] = float(np.clip(floor, LV3_MULTIPLIER_MIN, LV3_MULTIPLIER_MAX))
    return floors


def _event_floor_for_row(row: pd.Series, model: SpikeMultiplierModel) -> float:
    floor = 0.0
    for col, value in model.event_floor_multipliers.items():
        if float(row.get(col, 0.0) or 0.0) > 0.0:
            floor = max(floor, value)
    return floor


def _activity_fill_values(train: pd.DataFrame) -> Dict[str, float]:
    values: Dict[str, float] = {}
    zero_fill_cols = PLANNED_PROMO_COLUMNS | {
        "revenue_shock_7d",
        "cogs_shock_7d",
    }
    for col in SPIKE_ACTIVITY_COLUMNS:
        if col not in train.columns:
            continue
        if col in zero_fill_cols:
            values[col] = 0.0
            continue
        series = train[col].replace([np.inf, -np.inf], np.nan)
        positive = series[series > 0]
        values[col] = float(positive.median()) if len(positive) else float(series.median(skipna=True) or 0.0)
    return values


def _target_history_columns(target_col: str) -> set[str]:
    columns = {f"{target_col}_lag_{lag}d" for lag in TARGET_HISTORY_LAGS}
    columns.update(f"{target_col}_ma_{window}d" for window in TARGET_HISTORY_WINDOWS)
    columns.add(f"{target_col}_shock_7d")
    return columns


def _history_map(daily: pd.DataFrame, target_col: str, end_date: pd.Timestamp) -> Dict[pd.Timestamp, float]:
    hist = daily[(daily["date"] <= end_date) & daily[target_col].notna()][["date", target_col]].copy()
    return {
        pd.Timestamp(row.date).normalize(): float(getattr(row, target_col))
        for row in hist.itertuples(index=False)
    }


def _target_history_features_for_date(
    date: pd.Timestamp,
    values_by_date: Dict[pd.Timestamp, float],
    target_col: str,
    fallback_values: Dict[str, float],
) -> Dict[str, float]:
    key = pd.Timestamp(date).normalize()
    features: Dict[str, float] = {}
    for lag in TARGET_HISTORY_LAGS:
        col = f"{target_col}_lag_{lag}d"
        features[col] = values_by_date.get(key - pd.Timedelta(days=lag), fallback_values.get(col, np.nan))
    for window in TARGET_HISTORY_WINDOWS:
        col = f"{target_col}_ma_{window}d"
        vals = [
            values_by_date.get(key - pd.Timedelta(days=offset), np.nan)
            for offset in range(1, window + 1)
        ]
        arr = np.asarray(vals, dtype=float)
        features[col] = float(np.nanmean(arr)) if np.isfinite(arr).any() else fallback_values.get(col, np.nan)
    lag1 = values_by_date.get(key - pd.Timedelta(days=1), np.nan)
    ma7 = features[f"{target_col}_ma_7d"]
    shock_col = f"{target_col}_shock_7d"
    if np.isfinite(lag1) and np.isfinite(ma7) and ma7 > EPS:
        features[shock_col] = float(np.clip(abs(lag1 - ma7) / ma7, 0.0, 5.0))
    else:
        features[shock_col] = fallback_values.get(shock_col, 0.0)
    return features


def _attach_autoregressive_target_history(
    df: pd.DataFrame,
    values_by_date: Dict[pd.Timestamp, float],
    target_col: str,
    fallback_values: Dict[str, float],
) -> pd.DataFrame:
    out = df.copy()
    rows = [
        _target_history_features_for_date(dt, values_by_date, target_col, fallback_values)
        for dt in pd.to_datetime(out["date"])
    ]
    hist = pd.DataFrame(rows, index=out.index)
    for col in hist.columns:
        out[col] = hist[col]
    return out


def _fill_unknown_future_activity(df: pd.DataFrame, model: SpikeMultiplierModel) -> pd.DataFrame:
    if model.train_end_date is None:
        return df
    out = df.copy()
    future_mask = pd.to_datetime(out["date"]).gt(model.train_end_date)
    target_history_cols = _target_history_columns(model.target_col)
    for col, fill_value in model.activity_fill_values.items():
        if col not in out.columns:
            continue
        if col in target_history_cols:
            continue
        if col in PLANNED_PROMO_COLUMNS:
            missing_future = future_mask & out[col].replace([np.inf, -np.inf], np.nan).isna()
            out.loc[missing_future, col] = 0.0
            continue
        missing_like = future_mask & out[col].replace([np.inf, -np.inf], np.nan).fillna(0.0).le(0.0)
        out.loc[missing_like, col] = fill_value
    return out


def fit_spike_multiplier_model(
    daily: pd.DataFrame,
    target_col: str,
    base_daily: pd.DataFrame,
    base_col: str,
    train_end_date: pd.Timestamp,
    train_start_date: Optional[pd.Timestamp] = None,
) -> SpikeMultiplierModel:
    train = daily[(daily["date"] <= train_end_date) & daily[target_col].notna()].copy()
    if train_start_date is not None:
        train = train[train["date"] >= train_start_date].copy()
    train = _prepare_base_join(train, base_daily, base_col)
    train = _attach_multiplier_base(train)
    train = train[train["lv3_multiplier_base_value"].notna() & train["lv3_multiplier_base_value"].gt(EPS)].copy()
    multiplier = train[target_col] / train["lv3_multiplier_base_value"]

    model = SpikeMultiplierModel(target_col=target_col)
    model.train_end_date = pd.Timestamp(train_end_date)
    model.activity_fill_values = _activity_fill_values(train)
    model.history_values = _history_map(daily, target_col, pd.Timestamp(train_end_date))
    model.event_floor_multipliers = _event_floor_multipliers(train, multiplier)
    model.fit(spike_features(train), multiplier)
    calibrate_lv3_interval_from_holdout(model, train, multiplier)
    return model


def apply_spike_multiplier(
    future_daily: pd.DataFrame,
    base_daily: pd.DataFrame,
    base_col: str,
    target_output_col: str,
    model: SpikeMultiplierModel,
) -> pd.DataFrame:
    raw_pred = _prepare_base_join(future_daily, base_daily, base_col)
    raw_pred = _attach_multiplier_base(raw_pred)
    pred = _fill_unknown_future_activity(raw_pred, model).sort_values("date").copy()
    values_by_date = dict(model.history_values)
    rows = []
    for _, row in pred.iterrows():
        one = pd.DataFrame([row])
        one = _attach_autoregressive_target_history(
            one,
            values_by_date,
            model.target_col,
            model.activity_fill_values,
        )
        features = spike_features(one)
        model_multiplier = float(model.predict_multiplier(features)[0])
        multiplier_cap = model.max_multiplier
        raw_multiplier = model_multiplier
        floor_multiplier = _event_floor_for_row(row, model)
        multiplier = float(np.clip(max(raw_multiplier, floor_multiplier), model.min_multiplier, model.max_multiplier))
        lo80, hi80 = model.interval_multiplier(features, coverage=0.80, dates=one["date"])
        lo90, hi90 = model.interval_multiplier(features, coverage=0.90, dates=one["date"])
        lo80_value = min(float(lo80[0]), multiplier_cap)
        lo90_value = min(float(lo90[0]), multiplier_cap)
        hi80_value = min(max(float(hi80[0]), multiplier), multiplier_cap)
        hi90_value = min(max(float(hi90[0]), multiplier), multiplier_cap)
        lv2_base = max(float(row["lv2_base_value"]), 0.0)
        multiplier_base = max(float(row.get("lv3_multiplier_base_value", lv2_base)), 0.0)
        pred_value = multiplier_base * multiplier
        values_by_date[pd.Timestamp(row["date"]).normalize()] = pred_value
        rows.append(
            {
                "date": row["date"],
                "week_id": row["week_id"],
                "week_start": row["week_start"],
                f"{target_output_col}_lv2_base": lv2_base,
                f"{target_output_col}_lv3_multiplier_base": multiplier_base,
                f"{target_output_col}_lv3_event_base_used": float(row.get("lv3_uses_weekly_avg_event_base", 0.0)),
                f"{target_output_col}_lv3_model_multiplier": model_multiplier,
                f"{target_output_col}_lv3_event_cap": multiplier_cap,
                f"{target_output_col}_lv3_raw_multiplier": raw_multiplier,
                f"{target_output_col}_lv3_backend": model.backend,
                f"{target_output_col}_event_floor_multiplier": floor_multiplier,
                f"{target_output_col}_event_floor_applied": float(floor_multiplier > (raw_multiplier + 1e-9)),
                f"{target_output_col}_before_floor": multiplier_base * model_multiplier,
                f"{target_output_col}_after_floor": pred_value,
                f"{target_output_col}_lv3_multiplier": multiplier,
                target_output_col: pred_value,
                f"{target_output_col}_p10": multiplier_base * lo80_value,
                f"{target_output_col}_p90": multiplier_base * hi80_value,
                f"{target_output_col}_p05": multiplier_base * lo90_value,
                f"{target_output_col}_p95": multiplier_base * hi90_value,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        order = {
            pd.Timestamp(dt).normalize(): pos
            for pos, dt in enumerate(pd.to_datetime(raw_pred["date"]))
        }
        out["_original_order"] = pd.to_datetime(out["date"]).map(order)
        out = out.sort_values("_original_order").drop(columns=["_original_order"]).reset_index(drop=True)
    for col in out.columns:
        if col not in {"date", "week_id", "week_start", f"{target_output_col}_lv3_backend"}:
            out[col] = out[col].clip(lower=0.0)
    return out
