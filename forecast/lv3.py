from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .common import EPS, covid_regime_flags, safe_div
from .lv1 import RidgeLogModel


LV3_MULTIPLIER_MIN = 0.25
LV3_MULTIPLIER_MAX = 4.00
LV3_INTERVAL_SCALE = 1.40

SPIKE_ACTIVITY_COLUMNS = [
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
    "return_rate",
    "refund_rate",
    "stockout_rate",
    "review_count",
    "avg_rating",
    "low_rating_share",
    "revenue_lag_7d",
    "revenue_lag_14d",
    "revenue_lag_28d",
    "revenue_ma_7d",
    "revenue_ma_28d",
    "cogs_lag_7d",
    "cogs_lag_14d",
    "cogs_lag_28d",
    "cogs_ma_7d",
    "cogs_ma_28d",
]


def _numeric_feature(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return df[col].replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)
    return pd.Series(default, index=df.index, dtype=float)


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

    out = pd.DataFrame(index=df.index)
    out["bias"] = 1.0
    lv2_base = _numeric_feature(df, "lv2_base_value")
    out["log_lv2_base"] = np.log1p(lv2_base)
    if "week_id" in df.columns:
        week_sum = lv2_base.groupby(df["week_id"]).transform("sum")
        week_mean = lv2_base.groupby(df["week_id"]).transform("mean")
        out["base_share_in_week"] = safe_div(lv2_base, week_sum).replace([np.inf, -np.inf], np.nan).fillna(1.0 / 7.0)
        out["base_vs_week_mean"] = safe_div(lv2_base, week_mean).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        out["base_rank_in_week"] = lv2_base.groupby(df["week_id"]).rank(pct=True).fillna(0.5)
    else:
        out["base_share_in_week"] = 1.0 / 7.0
        out["base_vs_week_mean"] = 1.0
        out["base_rank_in_week"] = 0.5
    for weekday in range(1, 8):
        out[f"wday_{weekday}"] = iso_weekday.eq(weekday).astype(float)
    for month_id in range(1, 13):
        out[f"month_{month_id}"] = month.eq(month_id).astype(float)
    out["dom_sin"] = np.sin(2 * np.pi * day_of_month / 31.0)
    out["dom_cos"] = np.cos(2 * np.pi * day_of_month / 31.0)
    out["doy_sin"] = np.sin(2 * np.pi * day_of_year / 366.0)
    out["doy_cos"] = np.cos(2 * np.pi * day_of_year / 366.0)
    out["is_payday_window"] = _numeric_feature(df, "is_payday_window")
    out["is_holiday"] = _numeric_feature(df, "is_holiday")
    out["is_month_start"] = _numeric_feature(df, "is_month_start")
    out["is_month_end"] = _numeric_feature(df, "is_month_end")
    out["is_tet_window"] = _numeric_feature(df, "is_tet_window")
    out["is_black_friday_window"] = _numeric_feature(df, "is_black_friday_window")
    out["is_double_day_sale"] = (
        _numeric_feature(df, "is_double_day_sale")
        if "is_double_day_sale" in df.columns
        else ((month == day_of_month) & month.isin([9, 10, 11, 12])).astype(float)
    )
    out["is_1111_1212"] = (
        _numeric_feature(df, "is_1111_1212")
        if "is_1111_1212" in df.columns
        else (((month == 11) & (day_of_month == 11)) | ((month == 12) & (day_of_month == 12))).astype(float)
    )
    out["is_year_end_window"] = (
        _numeric_feature(df, "is_year_end_window")
        if "is_year_end_window" in df.columns
        else ((month == 12) & day_of_month.ge(15)).astype(float)
    )
    out["is_new_year_window"] = (
        _numeric_feature(df, "is_new_year_window")
        if "is_new_year_window" in df.columns
        else ((month == 1) & day_of_month.le(7)).astype(float)
    )

    for col in SPIKE_ACTIVITY_COLUMNS:
        raw = _numeric_feature(df, col)
        out[col] = raw
        out[f"log1p_{col}"] = np.log1p(raw.clip(lower=0.0))
    out["orders_per_session_safe"] = safe_div(_numeric_feature(df, "orders_count"), _numeric_feature(df, "sessions"))
    revenue_lag7 = _numeric_feature(df, "revenue_lag_7d")
    cogs_lag7 = _numeric_feature(df, "cogs_lag_7d")
    target_lag7 = revenue_lag7.where(revenue_lag7.gt(0), cogs_lag7)
    out["lag7_vs_base"] = safe_div(target_lag7, lv2_base).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 5.0)
    out["event_intensity"] = (
        out["is_holiday"]
        + out["is_tet_window"]
        + out["is_black_friday_window"]
        + out["is_double_day_sale"]
        + out["is_1111_1212"]
        + out["is_payday_window"]
        + _numeric_feature(df, "active_promo_count").clip(lower=0.0, upper=3.0)
    )
    return out


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
        self.model: Optional[RidgeLogModel] = None
        self.up_classifier: Optional[RidgeLogModel] = None
        self.down_classifier: Optional[RidgeLogModel] = None
        self.up_model: Optional[RidgeLogModel] = None
        self.down_model: Optional[RidgeLogModel] = None
        self.normal_model: Optional[RidgeLogModel] = None
        self.fallback_log_multiplier = 0.0
        self.fallback_up_log_multiplier = np.log(1.45)
        self.fallback_down_log_multiplier = np.log(0.70)
        self.abs_log_residual_q80 = 0.18
        self.abs_log_residual_q90 = 0.28
        self.train_end_date: Optional[pd.Timestamp] = None
        self.activity_fill_values: Dict[str, float] = {}
        self.up_threshold = 1.35
        self.down_threshold = 0.70

    def fit(self, X: pd.DataFrame, multiplier: pd.Series) -> "SpikeMultiplierModel":
        y = np.log(multiplier.clip(lower=self.min_multiplier, upper=self.max_multiplier))
        finite = y.replace([np.inf, -np.inf], np.nan).notna()
        X = X.loc[finite].copy()
        y = y.loc[finite]
        clean_multiplier = multiplier.loc[finite].replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(
            lower=self.min_multiplier,
            upper=self.max_multiplier,
        )
        if len(y):
            self.fallback_log_multiplier = float(np.median(y))
            self.up_threshold = float(max(1.25, np.nanquantile(clean_multiplier, 0.85)))
            self.down_threshold = float(min(0.80, np.nanquantile(clean_multiplier, 0.15)))
            up_vals = y.loc[clean_multiplier.ge(self.up_threshold)]
            down_vals = y.loc[clean_multiplier.le(self.down_threshold)]
            if len(up_vals):
                self.fallback_up_log_multiplier = float(np.median(up_vals))
            if len(down_vals):
                self.fallback_down_log_multiplier = float(np.median(down_vals))
        if len(y) >= 20:
            self.model = RidgeLogModel(alpha=self.alpha).fit(X, y)
            up_label = clean_multiplier.ge(self.up_threshold).astype(float)
            down_label = clean_multiplier.le(self.down_threshold).astype(float)
            if up_label.sum() >= 10 and up_label.sum() < len(up_label):
                self.up_classifier = RidgeLogModel(alpha=350.0).fit(X, up_label)
            if down_label.sum() >= 10 and down_label.sum() < len(down_label):
                self.down_classifier = RidgeLogModel(alpha=350.0).fit(X, down_label)
            normal_mask = ~(up_label.astype(bool) | down_label.astype(bool))
            if normal_mask.sum() >= 20:
                self.normal_model = RidgeLogModel(alpha=150.0).fit(X.loc[normal_mask], y.loc[normal_mask])
            if up_label.sum() >= 20:
                self.up_model = RidgeLogModel(alpha=180.0).fit(X.loc[up_label.astype(bool)], y.loc[up_label.astype(bool)])
            if down_label.sum() >= 20:
                self.down_model = RidgeLogModel(alpha=180.0).fit(X.loc[down_label.astype(bool)], y.loc[down_label.astype(bool)])
            pred = self.model.predict(X)
            residual = y.to_numpy() - pred
            residual = residual[np.isfinite(residual)]
            if len(residual):
                self.abs_log_residual_q80 = float(np.quantile(np.abs(residual), 0.80))
                self.abs_log_residual_q90 = float(np.quantile(np.abs(residual), 0.90))
        return self

    def _predict_log(self, model: Optional[RidgeLogModel], X: pd.DataFrame, fallback: float) -> np.ndarray:
        if model is None:
            return np.full(len(X), fallback, dtype=float)
        return np.asarray(model.predict(X), dtype=float)

    def _predict_probability(self, model: Optional[RidgeLogModel], X: pd.DataFrame) -> np.ndarray:
        if model is None:
            return np.zeros(len(X), dtype=float)
        return np.asarray(model.predict(X), dtype=float).clip(0.0, 0.45)

    def predict_multiplier(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            pred_log = np.full(len(X), self.fallback_log_multiplier, dtype=float)
        else:
            pred_log = np.asarray(self.model.predict(X), dtype=float)
        base = np.exp(pred_log).clip(self.min_multiplier, self.max_multiplier)
        p_up = self._predict_probability(self.up_classifier, X)
        p_down = self._predict_probability(self.down_classifier, X)
        total_p = p_up + p_down
        too_high = total_p > 0.55
        p_up = np.where(too_high, p_up / np.maximum(total_p, EPS) * 0.55, p_up)
        p_down = np.where(too_high, p_down / np.maximum(total_p, EPS) * 0.55, p_down)

        up = np.exp(self._predict_log(self.up_model, X, self.fallback_up_log_multiplier)).clip(
            self.up_threshold,
            self.max_multiplier,
        )
        down = np.exp(self._predict_log(self.down_model, X, self.fallback_down_log_multiplier)).clip(
            self.min_multiplier,
            self.down_threshold,
        )
        normal = np.exp(self._predict_log(self.normal_model, X, self.fallback_log_multiplier)).clip(
            self.min_multiplier,
            self.max_multiplier,
        )
        hybrid = p_up * up + p_down * down + np.maximum(1.0 - p_up - p_down, 0.0) * normal
        return (0.70 * base + 0.30 * hybrid).clip(self.min_multiplier, self.max_multiplier)

    def interval_multiplier(self, X: pd.DataFrame, coverage: float = 0.80) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            pred_log = np.full(len(X), self.fallback_log_multiplier, dtype=float)
        else:
            pred_log = np.asarray(self.model.predict(X), dtype=float)
        q = (self.abs_log_residual_q80 if coverage <= 0.80 else self.abs_log_residual_q90) * LV3_INTERVAL_SCALE
        lo = np.exp(pred_log - q).clip(self.min_multiplier, self.max_multiplier)
        hi = np.exp(pred_log + q).clip(self.min_multiplier, self.max_multiplier)
        return lo, hi


def _prepare_base_join(
    daily: pd.DataFrame,
    base_daily: pd.DataFrame,
    base_col: str,
) -> pd.DataFrame:
    out = daily.merge(base_daily[["date", base_col]], on="date", how="left")
    out = out.rename(columns={base_col: "lv2_base_value"})
    return out


def _activity_fill_values(train: pd.DataFrame) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for col in SPIKE_ACTIVITY_COLUMNS:
        if col not in train.columns:
            continue
        series = train[col].replace([np.inf, -np.inf], np.nan)
        positive = series[series > 0]
        values[col] = float(positive.median()) if len(positive) else float(series.median(skipna=True) or 0.0)
    return values


def _fill_unknown_future_activity(df: pd.DataFrame, model: SpikeMultiplierModel) -> pd.DataFrame:
    if model.train_end_date is None:
        return df
    out = df.copy()
    future_mask = pd.to_datetime(out["date"]).gt(model.train_end_date)
    for col, fill_value in model.activity_fill_values.items():
        if col not in out.columns:
            continue
        missing_like = future_mask & out[col].replace([np.inf, -np.inf], np.nan).fillna(0.0).le(0.0)
        out.loc[missing_like, col] = fill_value
    return out


def tactical_event_floor(df: pd.DataFrame, target_output_col: str) -> np.ndarray:
    floor = np.ones(len(df), dtype=float)
    is_holiday = _numeric_feature(df, "is_holiday").gt(0).to_numpy()
    is_tet = _numeric_feature(df, "is_tet_window").gt(0).to_numpy()
    is_black_friday = _numeric_feature(df, "is_black_friday_window").gt(0).to_numpy()
    is_double_day = _numeric_feature(df, "is_double_day_sale").gt(0).to_numpy()
    is_1111_1212 = _numeric_feature(df, "is_1111_1212").gt(0).to_numpy()
    is_year_end = _numeric_feature(df, "is_year_end_window").gt(0).to_numpy()
    is_new_year = _numeric_feature(df, "is_new_year_window").gt(0).to_numpy()
    has_promo = _numeric_feature(df, "active_promo_count").gt(0).to_numpy()
    strong_discount = _numeric_feature(df, "avg_promo_discount_value").ge(15).to_numpy()

    if target_output_col == "Revenue":
        floor = np.where(is_holiday, np.maximum(floor, 1.10), floor)
        floor = np.where(is_new_year, np.maximum(floor, 1.18), floor)
        floor = np.where(is_tet, np.maximum(floor, 1.25), floor)
        floor = np.where(is_year_end, np.maximum(floor, 1.25), floor)
        floor = np.where(is_double_day, np.maximum(floor, 1.35), floor)
        floor = np.where(is_black_friday, np.maximum(floor, 1.45), floor)
        floor = np.where(is_1111_1212, np.maximum(floor, 1.65), floor)
        floor = np.where(has_promo, np.maximum(floor, 1.12), floor)
        floor = np.where(has_promo & strong_discount, np.maximum(floor, 1.25), floor)
    else:
        floor = np.where(is_holiday, np.maximum(floor, 1.08), floor)
        floor = np.where(is_new_year, np.maximum(floor, 1.15), floor)
        floor = np.where(is_tet, np.maximum(floor, 1.22), floor)
        floor = np.where(is_year_end, np.maximum(floor, 1.22), floor)
        floor = np.where(is_double_day, np.maximum(floor, 1.30), floor)
        floor = np.where(is_black_friday, np.maximum(floor, 1.38), floor)
        floor = np.where(is_1111_1212, np.maximum(floor, 1.55), floor)
        floor = np.where(has_promo, np.maximum(floor, 1.10), floor)
        floor = np.where(has_promo & strong_discount, np.maximum(floor, 1.20), floor)
    return floor


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
    train = train[train["lv2_base_value"].notna() & train["lv2_base_value"].gt(EPS)].copy()
    multiplier = train[target_col] / train["lv2_base_value"]

    model = SpikeMultiplierModel(target_col=target_col)
    model.train_end_date = pd.Timestamp(train_end_date)
    model.activity_fill_values = _activity_fill_values(train)
    model.fit(spike_features(train), multiplier)
    return model


def apply_spike_multiplier(
    future_daily: pd.DataFrame,
    base_daily: pd.DataFrame,
    base_col: str,
    target_output_col: str,
    model: SpikeMultiplierModel,
) -> pd.DataFrame:
    raw_pred = _prepare_base_join(future_daily, base_daily, base_col)
    pred = _fill_unknown_future_activity(raw_pred, model)
    features = spike_features(pred)
    multiplier = model.predict_multiplier(features)
    multiplier = np.maximum(multiplier, tactical_event_floor(raw_pred, target_output_col)).clip(
        model.min_multiplier,
        model.max_multiplier,
    )
    lo80, hi80 = model.interval_multiplier(features, coverage=0.80)
    lo90, hi90 = model.interval_multiplier(features, coverage=0.90)
    hi80 = np.maximum(hi80, multiplier)
    hi90 = np.maximum(hi90, multiplier)

    out = pred[["date", "week_id", "week_start", "lv2_base_value"]].copy()
    out[f"{target_output_col}_lv2_base"] = out["lv2_base_value"].clip(lower=0.0)
    out[f"{target_output_col}_lv3_multiplier"] = multiplier
    out[target_output_col] = out[f"{target_output_col}_lv2_base"] * out[f"{target_output_col}_lv3_multiplier"]
    out[f"{target_output_col}_p10"] = out[f"{target_output_col}_lv2_base"] * lo80
    out[f"{target_output_col}_p90"] = out[f"{target_output_col}_lv2_base"] * hi80
    out[f"{target_output_col}_p05"] = out[f"{target_output_col}_lv2_base"] * lo90
    out[f"{target_output_col}_p95"] = out[f"{target_output_col}_lv2_base"] * hi90
    out = out.drop(columns=["lv2_base_value"])
    for col in out.columns:
        if col not in {"date", "week_id", "week_start"}:
            out[col] = out[col].clip(lower=0.0)
    return out
