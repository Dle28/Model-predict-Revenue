from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .common import (
    COVID_DROP_END_WEEK_ID,
    COVID_DROP_START_WEEK_ID,
    EPS,
    NORMALIZATION_START_WEEK_ID,
    RECOVERY_START_WEEK_ID,
    covid_regime_flag_frame,
)
from .intervals import split_conformal_abs_quantile


DEFAULT_STRUCTURAL_WEEKLY_WEIGHT = 0.0
STRUCTURAL_INTERVAL_HOLDOUT_WEEKS = 52
STRUCTURAL_RECENCY_HALFLIFE_WEEKS = 260.0
STRUCTURAL_INTERVAL_SCALE = 1.15


def requested_structural_weekly_weight(default: float = DEFAULT_STRUCTURAL_WEEKLY_WEIGHT) -> float:
    raw = os.environ.get("FORECAST_STRUCTURAL_WEEKLY_WEIGHT", "").strip()
    if not raw:
        return default
    try:
        return float(np.clip(float(raw), 0.0, 0.80))
    except ValueError:
        return default


def _numeric_feature(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return df[col].replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def structural_weekly_features(weekly: pd.DataFrame) -> pd.DataFrame:
    week_start = pd.to_datetime(weekly["week_start"])
    time_index = _numeric_feature(weekly, "time_index_week")
    origin = float(time_index.min()) if len(time_index) else 0.0
    time_years = (time_index - origin) / 52.0
    iso_week = _numeric_feature(weekly, "iso_week", 1.0)
    month = _numeric_feature(weekly, "month", 1.0)

    out = pd.DataFrame(index=weekly.index)
    out["time_years"] = time_years
    out["time_years_sq"] = np.square(time_years)
    out["time_years_sqrt"] = np.sqrt(time_years.clip(lower=0.0))
    for k in range(1, 4):
        out[f"week_sin_{k}"] = np.sin(2 * np.pi * k * iso_week / 53.0)
        out[f"week_cos_{k}"] = np.cos(2 * np.pi * k * iso_week / 53.0)
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    regimes = covid_regime_flag_frame(weekly["week_id"].astype(str), week_start)
    out["pre_covid"] = regimes["pre_covid"].astype(float)

    for col in [
        "is_month_start_week",
        "is_month_end_week",
        "is_payday_week",
        "is_tet_like_period",
        "is_black_friday_like_period",
        "promo_has_active",
        "active_promo_count",
        "avg_promo_discount_value",
        "stackable_promo_count",
        "promo_start_days",
        "promo_end_days",
        "expected_orders_per_1000_sessions",
        "orders_per_session_lag_14d",
        "revenue_per_session_lag_28d",
        "funnel_efficiency_lag_28d",
        "expected_cod_order_share",
        "expected_stockout_rate",
        "expected_fill_rate",
        "expected_top_product_revenue_share",
        "expected_streetwear_concentration_risk",
        "streetwear_concentration_risk",
        "expected_promo_discount_rate",
        "expected_promo_order_share",
        "expected_lost_sales_index",
        "promo_margin_pressure",
        "is_operational_crisis",
        "internal_stress_regime",
    ]:
        out[col] = _numeric_feature(weekly, col)
    out["log1p_active_promo_count"] = np.log1p(out["active_promo_count"].clip(lower=0.0))
    out["log1p_avg_promo_discount_value"] = np.log1p(out["avg_promo_discount_value"].clip(lower=0.0))
    out["log1p_expected_orders_per_1000_sessions"] = np.log1p(out["expected_orders_per_1000_sessions"].clip(lower=0.0))
    return out


def _structural_sample_weight(weekly: pd.DataFrame, train_end_pos: int) -> pd.Series:
    week_id = weekly["week_id"].astype(str)
    time_index = _numeric_feature(weekly, "time_index_week")
    max_time = float(time_index.loc[weekly.index <= train_end_pos].max()) if len(time_index) else 0.0
    age_weeks = (max_time - time_index).clip(lower=0.0)
    recency = np.power(0.5, age_weeks / max(STRUCTURAL_RECENCY_HALFLIFE_WEEKS, EPS))
    weight = pd.Series(recency, index=weekly.index, dtype=float)
    weight = weight.mask(week_id.between(COVID_DROP_START_WEEK_ID, COVID_DROP_END_WEEK_ID), weight * 0.60)
    recovery_mask = week_id.ge(RECOVERY_START_WEEK_ID) & week_id.lt(NORMALIZATION_START_WEEK_ID)
    weight = weight.mask(recovery_mask, weight * 1.35)
    return weight.clip(lower=0.05, upper=2.0)


@dataclass
class StructuralWeeklyModel:
    target_col: str
    alpha: float = 60.0
    features: Optional[List[str]] = None
    medians: Optional[pd.Series] = None
    means: Optional[pd.Series] = None
    stds: Optional[pd.Series] = None
    coef: Optional[np.ndarray] = None
    abs_residual_q80: float = 0.25
    abs_residual_q90: float = 0.35

    def fit(
        self,
        X: pd.DataFrame,
        y_log: pd.Series,
        sample_weight: Optional[pd.Series | np.ndarray] = None,
    ) -> "StructuralWeeklyModel":
        if len(X) < 20:
            raise ValueError(f"Need at least 20 structural training rows, got {len(X)}")
        self.features = list(X.columns)
        x = X.replace([np.inf, -np.inf], np.nan).astype(float)
        self.medians = x.median().fillna(0.0)
        x = x.fillna(self.medians)
        self.means = x.mean()
        self.stds = x.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        xs = (x - self.means) / self.stds
        xa = np.column_stack([np.ones(len(xs)), xs.to_numpy()])
        ya = y_log.astype(float).to_numpy()
        if sample_weight is not None:
            weight = np.asarray(sample_weight, dtype=float)
            weight = np.where(np.isfinite(weight) & (weight > 0.0), weight, 1.0)
            weight = np.sqrt(weight)
            xa = xa * weight[:, None]
            ya = ya * weight
        penalty = np.eye(xa.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        lhs = xa.T @ xa + penalty
        rhs = xa.T @ ya
        try:
            self.coef = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            self.coef = np.linalg.pinv(lhs) @ rhs
        residual = y_log.to_numpy(dtype=float) - self.predict_log(X)
        self.set_interval_residuals(residual)
        return self

    def _clean(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.features is None or self.medians is None or self.means is None or self.stds is None:
            raise RuntimeError("Structural model is not fitted")
        x = X.reindex(columns=self.features).replace([np.inf, -np.inf], np.nan).astype(float)
        return x.fillna(self.medians)

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("Structural model is not fitted")
        x = self._clean(X)
        xs = (x - self.means) / self.stds
        xa = np.column_stack([np.ones(len(xs)), xs.to_numpy()])
        return xa @ self.coef

    def predict_value(self, X: pd.DataFrame) -> np.ndarray:
        return np.expm1(self.predict_log(X)).clip(min=0.0)

    def set_interval_residuals(self, residual: pd.Series | np.ndarray) -> None:
        values = np.asarray(residual, dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            self.abs_residual_q80 = split_conformal_abs_quantile(values, 0.80, self.abs_residual_q80)
            self.abs_residual_q90 = split_conformal_abs_quantile(values, 0.90, self.abs_residual_q90)

    def interval(self, pred_log: float, horizon: int, coverage: float = 0.80) -> tuple[float, float]:
        base_q = self.abs_residual_q80 if coverage <= 0.80 else self.abs_residual_q90
        capped_horizon = min(max(horizon - 1, 0), 156)
        horizon_scale = 1.0 + math.sqrt(capped_horizon / 156.0)
        q = base_q * horizon_scale * STRUCTURAL_INTERVAL_SCALE
        return max(float(np.expm1(pred_log - q)), 0.0), max(float(np.expm1(pred_log + q)), 0.0)


def _training_mask(
    weekly: pd.DataFrame,
    target_col: str,
    train_end_week_id: str,
    train_start_week_id: str | None,
) -> pd.Series:
    mask = weekly["week_id"].astype(str).le(str(train_end_week_id))
    if train_start_week_id is not None:
        mask = mask & weekly["week_id"].astype(str).ge(str(train_start_week_id))
    complete = weekly.get("complete_target_week", True)
    return mask & complete & weekly[target_col].notna()


def fit_structural_weekly_model(
    weekly: pd.DataFrame,
    target_col: str,
    train_end_week_id: str,
    train_start_week_id: str | None = None,
) -> StructuralWeeklyModel:
    train_mask = _training_mask(weekly, target_col, train_end_week_id, train_start_week_id)
    features = structural_weekly_features(weekly)
    y = np.log1p(weekly[target_col].replace([np.inf, -np.inf], np.nan).astype(float))
    usable = train_mask & y.notna()
    eligible = weekly.index[weekly["week_id"].le(train_end_week_id)]
    train_end_pos = int(eligible.max()) if len(eligible) else int(weekly.index.min())
    sample_weight = _structural_sample_weight(weekly, train_end_pos).loc[usable]
    model = StructuralWeeklyModel(target_col=target_col).fit(features.loc[usable], y.loc[usable], sample_weight)

    usable_index = features.loc[usable].index
    val_size = min(STRUCTURAL_INTERVAL_HOLDOUT_WEEKS, max(8, len(usable_index) // 5))
    if len(usable_index) >= val_size + 30:
        train_index = usable_index[:-val_size]
        val_index = usable_index[-val_size:]
        try:
            holdout = StructuralWeeklyModel(target_col=target_col).fit(
                features.loc[train_index],
                y.loc[train_index],
                _structural_sample_weight(weekly, int(train_index.max())).loc[train_index],
            )
            residual = y.loc[val_index].to_numpy(dtype=float) - holdout.predict_log(features.loc[val_index])
            model.set_interval_residuals(residual)
        except Exception:
            pass
    return model
