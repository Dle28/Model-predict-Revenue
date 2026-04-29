from __future__ import annotations

import math
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .common import (
    COVID_DROP_END_WEEK_ID,
    COVID_DROP_START_WEEK_ID,
    EPS,
    NORMALIZATION_START_WEEK_ID,
    PRE_COVID_END_WEEK_ID,
    RECOVERY_END_WEEK_ID,
    RECOVERY_START_WEEK_ID,
    covid_regime_flags,
    safe_div,
)


TARGET_LAGS = [1, 2, 4, 8, 13, 26, 52]
TARGET_WINDOWS = [4, 8, 13, 26]
TARGET_VOL_WINDOWS = [4, 8, 13]
EXOGENOUS_LAGS = [1, 2]
EXOGENOUS_WINDOWS = [4]
EXOGENOUS_GROWTH_COLUMNS = {
    "sessions",
    "unique_visitors",
    "orders_count",
    "active_customers",
    "new_customers",
    "orders_per_session",
    "orders_per_customer",
    "aov",
    "discount_rate",
    "promo_intensity",
    "stockout_rate",
    "fill_rate",
    "return_rate",
    "refund_rate",
    "avg_rating",
    "cogs_ratio_w",
    "gross_margin_rate",
}
SMOOTH_WINDOW_WEEKS = 5
INTERVAL_RESIDUAL_SCALE = 1.50
LV1_EXOGENOUS_BASE_COLUMNS = [
    "sessions",
    "unique_visitors",
    "pageviews_per_session",
    "source_entropy",
    "orders_count",
    "active_customers",
    "new_customers",
    "new_order_customers",
    "repeat_customers",
    "orders_per_session",
    "orders_per_customer",
    "new_customer_ratio",
    "repeat_customer_ratio",
    "cancel_rate",
    "fulfilled_rate",
    "aov",
    "items_per_order",
    "avg_unit_price",
    "discount_amount_per_order",
    "discount_rate",
    "promo_intensity",
    "promo_order_share",
    "stacked_promo_share",
    "premium_share",
    "standard_share",
    "stockout_rate",
    "stockout_days",
    "fill_rate",
    "days_of_supply",
    "overstock_rate",
    "reorder_rate",
    "sell_through_rate",
    "stock_pressure_index",
    "available_supply_index",
    "return_rate",
    "refund_rate",
    "defective_ratio",
    "wrong_size_ratio",
    "not_as_described_ratio",
    "avg_rating",
    "review_count",
    "low_rating_share",
    "has_holiday",
    "is_month_start_week",
    "is_month_end_week",
    "is_payday_week",
    "is_tet_like_period",
    "is_black_friday_like_period",
    "promo_has_active",
    "promo_active_days",
    "promo_start_days",
    "promo_end_days",
    "promo_discount_sum",
    "avg_promo_discount_value",
    "promo_intensity_day",
    "stackable_promo_count",
    "cogs_ratio_w",
    "gross_margin_rate",
]
LV1_EXOGENOUS_PREFIXES: List[str] = []


MODEL_BACKEND_ALIASES = {
    "lgb": "lightgbm",
    "lgbm": "lightgbm",
    "xgb": "xgboost",
}
MODEL_BACKENDS = {"ridge", "lightgbm", "xgboost", "auto"}
DEFAULT_MODEL_BACKEND = "ridge"
LV1_TARGET_MODES = {"smooth", "raw", "raw_plus_smooth_lag"}
DEFAULT_LV1_TARGET_MODE = "raw"
PRECOVID_BASELINE_START_YEAR = 2015
PRECOVID_BASELINE_END_YEAR = 2018
PRECOVID_BASELINE_QUANTILE = 0.60


LV1_BASE_WEEKLY_FEATURES = [
    "time_index_week",
    "iso_week",
    "month",
    "quarter",
    "week_sin",
    "week_cos",
    "month_sin",
    "month_cos",
    "is_month_start_week",
    "is_month_end_week",
    "is_payday_week",
    "is_tet_like_period",
    "is_black_friday_like_period",
    "pre_covid",
    "covid_drop",
    "recovery_phase",
    "normalization_phase",
    "weeks_since_covid_start",
    "weeks_since_recovery_start",
    "recovery_progress",
    "pre_covid_baseline_same_week",
    "covid_drop_factor",
    "covid_adjusted_lag_52w",
    "recovery_safe_lag_52w",
    "lag52_to_precovid_ratio",
    "recovery_ratio",
    "recovery_gap",
    "pre_covid_gap",
    "expected_rebound_level",
    "target_growth_4w",
    "target_lag_52w_x_covid_drop",
    "target_lag_52w_x_recovery_phase",
    "covid_adjusted_lag_52w_x_recovery_phase",
    "pre_covid_baseline_x_recovery_progress",
    "recovery_gap_x_recovery_progress",
    "recovery_progress_x_month_sin",
    "recovery_progress_x_month_cos",
    "recovery_uplift_signal",
    "recovery_momentum",
    "target_level_shift_ratio",
    "target_growth_1w",
    "target_same_iso_week_1y",
    "target_same_iso_week_2y",
    "target_yoy_hist",
    "target_yoy_growth",
] + [f"target_lag_{lag}w" for lag in TARGET_LAGS] + [f"target_ma_{window}w" for window in TARGET_WINDOWS] + [
    f"target_vol_{window}w" for window in TARGET_VOL_WINDOWS
]


def lv1_exogenous_columns(weekly: pd.DataFrame) -> List[str]:
    columns = [col for col in LV1_EXOGENOUS_BASE_COLUMNS if col in weekly.columns]
    for prefix in LV1_EXOGENOUS_PREFIXES:
        columns.extend([col for col in weekly.columns if col.startswith(prefix)])
    seen: set[str] = set()
    deduped = []
    for col in columns:
        if col not in seen:
            seen.add(col)
            deduped.append(col)
    return deduped


def lv1_feature_columns(exog_columns: Iterable[str]) -> List[str]:
    feature_cols = list(LV1_BASE_WEEKLY_FEATURES)
    for col in exog_columns:
        for lag in EXOGENOUS_LAGS:
            feature_cols.append(f"{col}_lag_{lag}w")
        for window in EXOGENOUS_WINDOWS:
            feature_cols.append(f"{col}_ma_{window}w")
        if col in EXOGENOUS_GROWTH_COLUMNS:
            feature_cols.append(f"{col}_growth_1w")
            feature_cols.append(f"{col}_vs_ma4")
    return feature_cols


class RidgeLogModel:
    def __init__(self, alpha: float = 25.0) -> None:
        self.alpha = alpha
        self.features: List[str] = []
        self.medians: Optional[pd.Series] = None
        self.means: Optional[pd.Series] = None
        self.stds: Optional[pd.Series] = None
        self.coef: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series | np.ndarray] = None) -> "RidgeLogModel":
        if len(X) < 20:
            raise ValueError(f"Need at least 20 training rows, got {len(X)}")
        self.features = list(X.columns)
        x = X.replace([np.inf, -np.inf], np.nan).astype(float)
        self.medians = x.median().fillna(0.0)
        x = x.fillna(self.medians)
        self.means = x.mean()
        self.stds = x.std(ddof=0).replace(0, 1.0).fillna(1.0)
        xs = (x - self.means) / self.stds
        xa = np.column_stack([np.ones(len(xs)), xs.to_numpy()])
        ya = y.astype(float).to_numpy()
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
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef is None or self.medians is None or self.means is None or self.stds is None:
            raise RuntimeError("Model is not fitted")
        x = X.reindex(columns=self.features).replace([np.inf, -np.inf], np.nan).astype(float)
        x = x.fillna(self.medians)
        xs = (x - self.means) / self.stds
        xa = np.column_stack([np.ones(len(xs)), xs.to_numpy()])
        return xa @ self.coef


class WeeklyBaseLogModel:
    def __init__(self, alpha: float = 35.0) -> None:
        self.alpha = alpha
        self.backend = "ridge"
        self.features: List[str] = []
        self.medians: Optional[pd.Series] = None
        self.model: Optional[Any] = None
        self.ridge: Optional[RidgeLogModel] = None
        self.log_adjustment = 0.0
        self.abs_residual_q80 = 0.25
        self.abs_residual_q90 = 0.35

    def fit(
        self,
        X: pd.DataFrame,
        y_log: pd.Series,
        sample_weight: Optional[pd.Series | np.ndarray] = None,
    ) -> "WeeklyBaseLogModel":
        if len(X) < 20:
            raise ValueError(f"Need at least 20 training rows, got {len(X)}")
        self.features = list(X.columns)
        x = X.replace([np.inf, -np.inf], np.nan).astype(float)
        self.medians = x.median().fillna(0.0)
        x = x.fillna(self.medians)
        y = y_log.astype(float)
        clean_weight = None
        if sample_weight is not None:
            clean_weight = np.asarray(sample_weight, dtype=float)
            clean_weight = np.where(np.isfinite(clean_weight) & (clean_weight > 0.0), clean_weight, 1.0)

        self.model = None
        self.ridge = None
        requested_backend = requested_model_backend()
        fitted = False

        if requested_backend in {"lightgbm", "auto"}:
            try:
                import lightgbm as lgb  # type: ignore

                self.model = lgb.LGBMRegressor(
                    objective="regression_l1",
                    learning_rate=0.035,
                    n_estimators=650,
                    num_leaves=23,
                    min_child_samples=18,
                    subsample=0.85,
                    subsample_freq=1,
                    colsample_bytree=0.85,
                    reg_alpha=0.05,
                    reg_lambda=2.0,
                    random_state=42,
                    verbose=-1,
                )
                self.model.fit(x, y, sample_weight=clean_weight)
                self.backend = "lightgbm"
                fitted = True
            except Exception:
                self.model = None

        if not fitted and requested_backend in {"xgboost", "auto"}:
            try:
                import xgboost as xgb  # type: ignore

                self.model = xgb.XGBRegressor(
                    objective="reg:absoluteerror",
                    learning_rate=0.035,
                    n_estimators=650,
                    max_depth=4,
                    min_child_weight=8,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.05,
                    reg_lambda=2.0,
                    random_state=42,
                )
                self.model.fit(x, y, sample_weight=clean_weight)
                self.backend = "xgboost"
                fitted = True
            except Exception:
                self.model = None

        if not fitted:
            self.ridge = RidgeLogModel(alpha=self.alpha).fit(x, y, sample_weight=clean_weight)
            self.backend = "ridge"

        raw = self._predict_raw(x)
        residual = y.to_numpy() - raw
        residual = residual[np.isfinite(residual)]
        if len(residual):
            self.log_adjustment = float(np.clip(np.mean(residual) + 0.5 * np.var(residual), -0.35, 0.35))
            self.abs_residual_q80 = float(np.quantile(np.abs(residual), 0.80))
            self.abs_residual_q90 = float(np.quantile(np.abs(residual), 0.90))
        return self

    def _clean(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.medians is None:
            raise RuntimeError("Model is not fitted")
        x = X.reindex(columns=self.features).replace([np.inf, -np.inf], np.nan).astype(float)
        return x.fillna(self.medians)

    def _predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        x = self._clean(X)
        if self.backend in {"lightgbm", "xgboost"} and self.model is not None:
            return np.asarray(self.model.predict(x), dtype=float)
        if self.ridge is not None:
            return np.asarray(self.ridge.predict(x), dtype=float)
        raise RuntimeError("Model is not fitted")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict_raw(X) + self.log_adjustment

    def interval(self, pred_log: float, horizon: int, coverage: float = 0.80) -> Tuple[float, float]:
        base_q = self.abs_residual_q80 if coverage <= 0.80 else self.abs_residual_q90
        horizon_scale = 1.0 + min(max(horizon - 1, 0), 156) / 156.0
        q = base_q * horizon_scale * INTERVAL_RESIDUAL_SCALE
        return max(float(np.expm1(pred_log - q)), 0.0), max(float(np.expm1(pred_log + q)), 0.0)


def requested_model_backend() -> str:
    backend = os.environ.get("FORECAST_MODEL_BACKEND", DEFAULT_MODEL_BACKEND).strip().lower()
    backend = MODEL_BACKEND_ALIASES.get(backend, backend)
    return backend if backend in MODEL_BACKENDS else DEFAULT_MODEL_BACKEND


def requested_lv1_target_mode() -> str:
    mode = os.environ.get("FORECAST_LV1_TARGET_MODE", DEFAULT_LV1_TARGET_MODE).strip().lower()
    mode = mode.replace("-", "_")
    return mode if mode in LV1_TARGET_MODES else DEFAULT_LV1_TARGET_MODE


def smooth_weekly_target(values: pd.Series, window: int = SMOOTH_WINDOW_WEEKS) -> pd.Series:
    clean = values.replace([np.inf, -np.inf], np.nan).astype(float)
    return clean.rolling(window=window, min_periods=1).mean()


def lv1_target_series(weekly: pd.DataFrame, target_col: str) -> pd.Series:
    target_without_holiday = f"{target_col}_lv1_target"
    if target_without_holiday in weekly.columns:
        return weekly[target_without_holiday].replace([np.inf, -np.inf], np.nan).astype(float)
    return weekly[target_col].replace([np.inf, -np.inf], np.nan).astype(float)


def lv1_training_target_and_feature_values(
    weekly: pd.DataFrame,
    target_col: str,
    known_mask: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    raw_values = lv1_target_series(weekly, target_col).where(known_mask)
    smooth_values = smooth_weekly_target(raw_values)
    mode = requested_lv1_target_mode()
    if mode == "raw":
        return raw_values, raw_values
    if mode == "raw_plus_smooth_lag":
        return raw_values, smooth_values
    return smooth_values, smooth_values


def training_week_mask(
    weekly: pd.DataFrame,
    train_end_week_id: str,
    train_start_week_id: Optional[str] = None,
) -> pd.Series:
    mask = weekly["week_id"].le(train_end_week_id)
    if train_start_week_id is not None:
        mask = mask & weekly["week_id"].ge(train_start_week_id)
    return mask


def make_value_maps(weekly: pd.DataFrame, values: pd.Series) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float]]:
    by_pos: Dict[int, float] = {}
    by_iso: Dict[Tuple[int, int], float] = {}
    for pos, row in weekly.iterrows():
        val = values.iloc[pos]
        if pd.notna(val):
            fval = float(val)
            by_pos[int(pos)] = fval
            by_iso[(int(row["iso_year"]), int(row["iso_week"]))] = fval
    return by_pos, by_iso


def make_exogenous_maps(
    weekly: pd.DataFrame,
    columns: Iterable[str],
    known_end_pos: int,
) -> Dict[str, Dict[str, Any]]:
    maps: Dict[str, Dict[str, Any]] = {}
    for col in columns:
        by_pos: Dict[int, float] = {}
        by_iso: Dict[Tuple[int, int], float] = {}
        by_week: Dict[int, List[float]] = {}
        if col in weekly.columns:
            for pos, row in weekly.iloc[: known_end_pos + 1].iterrows():
                val = row[col]
                if pd.notna(val) and np.isfinite(val):
                    fval = float(val)
                    by_pos[int(pos)] = fval
                    by_iso[(int(row["iso_year"]), int(row["iso_week"]))] = fval
                    by_week.setdefault(int(row["iso_week"]), []).append(fval)
        vals = list(by_pos.values())
        maps[col] = {
            "by_pos": by_pos,
            "by_iso": by_iso,
            "by_week": by_week,
            "fallback": float(np.nanmedian(vals)) if vals else 0.0,
        }
    return maps


def exogenous_value(
    weekly: pd.DataFrame,
    target_pos: int,
    col: str,
    exog_maps: Dict[str, Dict[str, Any]],
) -> float:
    col_map = exog_maps.get(col, {})
    by_pos: Dict[int, float] = col_map.get("by_pos", {})
    by_iso: Dict[Tuple[int, int], float] = col_map.get("by_iso", {})
    by_week: Dict[int, List[float]] = col_map.get("by_week", {})
    fallback = float(col_map.get("fallback", 0.0))
    if target_pos in by_pos:
        return by_pos[target_pos]
    if target_pos < 0 or target_pos >= len(weekly):
        return fallback

    row = weekly.iloc[target_pos]
    same_last_year = by_iso.get((int(row["iso_year"]) - 1, int(row["iso_week"])))
    if same_last_year is not None and np.isfinite(same_last_year):
        return float(same_last_year)

    same_week_history = by_week.get(int(row["iso_week"]), [])
    if same_week_history:
        return float(np.nanmedian(same_week_history[-5:]))

    recent = [val for pos, val in sorted(by_pos.items()) if pos < target_pos][-13:]
    if recent:
        return float(np.nanmean(recent))
    return fallback


def add_exogenous_history_features(
    features: Dict[str, float],
    weekly: pd.DataFrame,
    pos: int,
    col: str,
    exog_maps: Dict[str, Dict[str, Any]],
) -> None:
    lag_values: Dict[int, float] = {}
    for lag in EXOGENOUS_LAGS:
        val = exogenous_value(weekly, pos - lag, col, exog_maps)
        lag_values[lag] = val
        features[f"{col}_lag_{lag}w"] = val

    for window in EXOGENOUS_WINDOWS:
        vals = [exogenous_value(weekly, pos - lag, col, exog_maps) for lag in range(1, window + 1)]
        features[f"{col}_ma_{window}w"] = mean_finite(vals)

    lag1 = lag_values.get(1, np.nan)
    lag2 = lag_values.get(2, np.nan)
    ma4 = features.get(f"{col}_ma_4w", np.nan)
    if col in EXOGENOUS_GROWTH_COLUMNS:
        features[f"{col}_growth_1w"] = (
            np.clip(safe_div(lag1 - lag2, lag2), -5.0, 5.0)
            if pd.notna(lag1) and pd.notna(lag2)
            else np.nan
        )
        features[f"{col}_vs_ma4"] = (
            np.clip(safe_div(lag1, ma4) - 1.0, -5.0, 5.0)
            if pd.notna(lag1) and pd.notna(ma4)
            else np.nan
        )


def mean_finite(values: Iterable[float], fallback: float = np.nan) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else fallback


def std_finite(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanstd(arr)) if np.isfinite(arr).sum() >= 2 else np.nan


def median_finite(values: Iterable[float], fallback: float = np.nan) -> float:
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.nanmedian(finite)) if len(finite) else fallback


def quantile_finite(values: Iterable[float], q: float, fallback: float = np.nan) -> float:
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.nanquantile(finite, q)) if len(finite) else fallback


def _neighbor_iso_weeks(iso_week: int) -> List[int]:
    return sorted({((int(iso_week) - 2) % 53) + 1, int(iso_week), (int(iso_week) % 53) + 1})


def _is_covid_drop_week(week_id: str) -> bool:
    week_id = str(week_id)
    return COVID_DROP_START_WEEK_ID <= week_id <= COVID_DROP_END_WEEK_ID


def make_precovid_reference_maps(weekly: pd.DataFrame, values: pd.Series) -> Dict[str, Any]:
    frame = weekly[["iso_year", "iso_week", "week_id"]].copy()
    frame["value"] = values.replace([np.inf, -np.inf], np.nan).astype(float).to_numpy()
    finite = frame["value"].notna() & np.isfinite(frame["value"])

    pre = frame[finite & frame["iso_year"].between(PRECOVID_BASELINE_START_YEAR, PRECOVID_BASELINE_END_YEAR)].copy()
    all_finite = frame[finite].copy()
    fallback_pre = median_finite(pre["value"], median_finite(all_finite["value"], 0.0))
    baseline_same_week = (
        pre.groupby("iso_week")["value"].quantile(PRECOVID_BASELINE_QUANTILE).to_dict()
        if len(pre)
        else {}
    )

    baseline_by_week: Dict[int, float] = {}
    for week in range(1, 54):
        same_week = float(baseline_same_week.get(week, np.nan))
        neighbor = quantile_finite(
            pre.loc[pre["iso_week"].isin(_neighbor_iso_weeks(week)), "value"],
            PRECOVID_BASELINE_QUANTILE,
            fallback_pre,
        )
        if np.isfinite(same_week) and np.isfinite(neighbor):
            baseline = 0.6 * same_week + 0.4 * neighbor
        elif np.isfinite(same_week):
            baseline = same_week
        else:
            baseline = neighbor
        baseline_by_week[week] = max(float(baseline), 0.0) if np.isfinite(baseline) else max(float(fallback_pre), 0.0)

    covid = frame[finite & frame["week_id"].between(COVID_DROP_START_WEEK_ID, COVID_DROP_END_WEEK_ID)].copy()
    global_covid_median = median_finite(covid["value"], np.nan)
    global_drop = safe_div(global_covid_median, fallback_pre) if np.isfinite(global_covid_median) and fallback_pre > EPS else 1.0
    global_drop = float(np.clip(global_drop, 0.35, 1.05)) if np.isfinite(global_drop) else 1.0

    drop_factor_by_week: Dict[int, float] = {}
    for week in range(1, 54):
        covid_median = median_finite(covid.loc[covid["iso_week"].eq(week), "value"], np.nan)
        baseline = baseline_by_week.get(week, fallback_pre)
        if np.isfinite(covid_median) and baseline > EPS:
            drop_factor = safe_div(covid_median, baseline)
        else:
            drop_factor = global_drop
        drop_factor_by_week[week] = float(np.clip(drop_factor, 0.35, 1.05)) if np.isfinite(drop_factor) else 1.0

    return {
        "baseline_by_week": baseline_by_week,
        "drop_factor_by_week": drop_factor_by_week,
        "fallback_baseline": max(float(fallback_pre), 0.0) if np.isfinite(fallback_pre) else 0.0,
        "fallback_drop_factor": global_drop,
    }


def regime_features(row: pd.Series) -> Dict[str, float]:
    return covid_regime_flags(str(row["week_id"]), pd.Timestamp(row["week_start"]))


def row_lv1_weekly_features(
    weekly: pd.DataFrame,
    pos: int,
    values_by_pos: Dict[int, float],
    values_by_iso: Dict[Tuple[int, int], float],
    precovid_refs: Dict[str, Any],
    exog_maps: Dict[str, Dict[str, Any]],
    exog_columns: Iterable[str],
) -> Dict[str, float]:
    row = weekly.iloc[pos]
    features = {
        "time_index_week": float(row["time_index_week"]),
        "iso_week": float(row["iso_week"]),
        "month": float(row["month"]),
        "quarter": float(row["quarter"]),
        "week_sin": float(row["week_sin"]),
        "week_cos": float(row["week_cos"]),
        "month_sin": float(row["month_sin"]),
        "month_cos": float(row["month_cos"]),
        "is_month_start_week": float(row.get("is_month_start_week", 0.0)),
        "is_month_end_week": float(row.get("is_month_end_week", 0.0)),
        "is_payday_week": float(row.get("is_payday_week", 0.0)),
        "is_tet_like_period": float(row.get("is_tet_like_period", 0.0)),
        "is_black_friday_like_period": float(row.get("is_black_friday_like_period", 0.0)),
    }
    features.update(regime_features(row))

    for lag in TARGET_LAGS:
        features[f"target_lag_{lag}w"] = values_by_pos.get(pos - lag, np.nan)

    iso_week = int(row["iso_week"])
    pre_covid_baseline = float(
        precovid_refs.get("baseline_by_week", {}).get(
            iso_week,
            precovid_refs.get("fallback_baseline", np.nan),
        )
    )
    covid_drop_factor = float(
        precovid_refs.get("drop_factor_by_week", {}).get(
            iso_week,
            precovid_refs.get("fallback_drop_factor", 1.0),
        )
    )
    covid_drop_factor = float(np.clip(covid_drop_factor, 0.35, 1.05)) if np.isfinite(covid_drop_factor) else 1.0
    lag52 = features["target_lag_52w"]
    lag52_week_id = str(weekly.iloc[pos - 52]["week_id"]) if pos >= 52 else ""
    if pd.notna(lag52) and np.isfinite(lag52):
        covid_adjusted_lag52 = float(lag52) / covid_drop_factor if _is_covid_drop_week(lag52_week_id) else float(lag52)
    else:
        covid_adjusted_lag52 = np.nan
    recovery_progress = float(features.get("recovery_progress", 0.0))
    if pd.notna(covid_adjusted_lag52) and np.isfinite(covid_adjusted_lag52) and np.isfinite(pre_covid_baseline):
        recovery_safe_lag52 = (1.0 - recovery_progress) * covid_adjusted_lag52 + recovery_progress * pre_covid_baseline
    elif np.isfinite(pre_covid_baseline):
        recovery_safe_lag52 = pre_covid_baseline
    else:
        recovery_safe_lag52 = covid_adjusted_lag52

    features["pre_covid_baseline_same_week"] = pre_covid_baseline
    features["covid_drop_factor"] = covid_drop_factor
    features["covid_adjusted_lag_52w"] = covid_adjusted_lag52
    features["recovery_safe_lag_52w"] = recovery_safe_lag52
    features["lag52_to_precovid_ratio"] = (
        safe_div(lag52, pre_covid_baseline)
        if pd.notna(lag52) and np.isfinite(lag52) and pre_covid_baseline > EPS
        else np.nan
    )

    for window in TARGET_WINDOWS:
        vals = [values_by_pos.get(pos - lag, np.nan) for lag in range(1, window + 1)]
        features[f"target_ma_{window}w"] = mean_finite(vals)
    for window in TARGET_VOL_WINDOWS:
        vals = [values_by_pos.get(pos - lag, np.nan) for lag in range(1, window + 1)]
        features[f"target_vol_{window}w"] = std_finite(vals)

    lag1 = features["target_lag_1w"]
    lag2 = features["target_lag_2w"]
    lag4 = features["target_lag_4w"]
    features["target_growth_1w"] = np.clip(safe_div(lag1 - lag2, lag2), -3.0, 3.0) if pd.notna(lag1) and pd.notna(lag2) else np.nan
    features["target_growth_4w"] = np.clip(safe_div(lag1 - lag4, lag4), -3.0, 3.0) if pd.notna(lag1) and pd.notna(lag4) else np.nan

    if pd.notna(lag1) and np.isfinite(lag1) and pre_covid_baseline > EPS:
        recovery_ratio = float(np.clip(safe_div(lag1, pre_covid_baseline), 0.3, 1.5))
        recovery_gap = float(np.clip(1.0 - recovery_ratio, -0.5, 0.7))
        pre_covid_gap = float(np.clip(safe_div(pre_covid_baseline - lag1, pre_covid_baseline), -0.5, 0.7))
        expected_rebound_level = float(lag1 + recovery_progress * max(pre_covid_baseline - float(lag1), 0.0))
    else:
        recovery_ratio = np.nan
        recovery_gap = np.nan
        pre_covid_gap = np.nan
        expected_rebound_level = np.nan
    features["recovery_ratio"] = recovery_ratio
    features["recovery_gap"] = recovery_gap
    features["pre_covid_gap"] = pre_covid_gap
    features["expected_rebound_level"] = expected_rebound_level
    features["target_lag_52w_x_covid_drop"] = lag52 * features.get("covid_drop", 0.0) if pd.notna(lag52) else np.nan
    features["target_lag_52w_x_recovery_phase"] = lag52 * features.get("recovery_phase", 0.0) if pd.notna(lag52) else np.nan
    features["covid_adjusted_lag_52w_x_recovery_phase"] = (
        covid_adjusted_lag52 * features.get("recovery_phase", 0.0) if pd.notna(covid_adjusted_lag52) else np.nan
    )
    features["pre_covid_baseline_x_recovery_progress"] = pre_covid_baseline * recovery_progress
    features["recovery_gap_x_recovery_progress"] = recovery_gap * recovery_progress if pd.notna(recovery_gap) else np.nan
    features["recovery_progress_x_month_sin"] = recovery_progress * features["month_sin"]
    features["recovery_progress_x_month_cos"] = recovery_progress * features["month_cos"]
    features["recovery_uplift_signal"] = (
        recovery_progress * (pre_covid_baseline - lag52)
        if pd.notna(lag52) and np.isfinite(pre_covid_baseline)
        else np.nan
    )
    features["recovery_momentum"] = (
        recovery_progress * features["target_growth_4w"]
        if pd.notna(features["target_growth_4w"])
        else np.nan
    )

    same_last_year = values_by_iso.get((int(row["iso_year"]) - 1, int(row["iso_week"])), np.nan)
    same_two_years = values_by_iso.get((int(row["iso_year"]) - 2, int(row["iso_week"])), np.nan)
    features["target_same_iso_week_1y"] = same_last_year
    features["target_same_iso_week_2y"] = same_two_years
    features["target_yoy_hist"] = np.clip(safe_div(lag1 - same_last_year, same_last_year), -3.0, 3.0) if pd.notna(lag1) and pd.notna(same_last_year) else np.nan
    features["target_yoy_growth"] = features["target_yoy_hist"]
    recent_13 = mean_finite([values_by_pos.get(pos - lag, np.nan) for lag in range(1, 14)])
    baseline_52 = mean_finite([values_by_pos.get(pos - lag, np.nan) for lag in range(1, 53)])
    features["target_level_shift_ratio"] = safe_div(recent_13, baseline_52) if pd.notna(recent_13) and pd.notna(baseline_52) else np.nan
    for col in exog_columns:
        add_exogenous_history_features(features, weekly, pos, col, exog_maps)
    return features


def build_lv1_weekly_feature_frame(weekly: pd.DataFrame, values: pd.Series, known_end_pos: int) -> pd.DataFrame:
    exog_columns = lv1_exogenous_columns(weekly)
    values_by_pos, values_by_iso = make_value_maps(weekly, values)
    precovid_refs = make_precovid_reference_maps(weekly, values)
    exog_maps = make_exogenous_maps(weekly, exog_columns, known_end_pos)
    rows = [
        row_lv1_weekly_features(weekly, pos, values_by_pos, values_by_iso, precovid_refs, exog_maps, exog_columns)
        for pos in range(len(weekly))
    ]
    return pd.DataFrame(rows, index=weekly.index).reindex(columns=lv1_feature_columns(exog_columns))


def train_end_position(weekly: pd.DataFrame, train_end_week_id: str) -> int:
    eligible = weekly.index[weekly["week_id"].le(train_end_week_id)]
    if len(eligible) == 0:
        return 0
    return int(eligible.max())


def estimate_same_iso_growth(
    weekly: pd.DataFrame,
    smooth_values: pd.Series,
    train_end_week_id: str,
    train_start_week_id: Optional[str] = None,
) -> float:
    train = weekly[training_week_mask(weekly, train_end_week_id, train_start_week_id)].copy()
    values = smooth_values.loc[train.index]
    by_iso = {
        (int(row["iso_year"]), int(row["iso_week"])): float(values.loc[idx])
        for idx, row in train.iterrows()
        if pd.notna(values.loc[idx])
    }
    ratios = []
    for idx, row in train.iterrows():
        val = values.loc[idx]
        prev = by_iso.get((int(row["iso_year"]) - 1, int(row["iso_week"])))
        if pd.notna(val) and prev is not None and prev > EPS:
            ratios.append(float(val) / prev)
    if not ratios:
        return 1.0
    return float(np.clip(np.median(ratios[-104:]), 0.7, 1.3))


def same_iso_reference_prediction(
    weekly: pd.DataFrame,
    pos: int,
    values_by_pos: Dict[int, float],
    values_by_iso: Dict[Tuple[int, int], float],
    growth: float,
    fallback_median: float,
) -> float:
    row = weekly.iloc[pos]
    same_last_year = values_by_iso.get((int(row["iso_year"]) - 1, int(row["iso_week"])))
    if same_last_year is not None and np.isfinite(same_last_year) and same_last_year > EPS:
        return max(float(same_last_year * growth), 0.0)

    recent = [values_by_pos.get(pos - lag, np.nan) for lag in range(1, 14)]
    recent = [float(v) for v in recent if pd.notna(v) and np.isfinite(v)]
    if recent:
        return max(float(np.mean(recent)), 0.0)
    return max(float(fallback_median), 0.0)


def dynamic_model_weight(feat: Dict[str, float]) -> float:
    base = 0.55
    shift = feat.get("target_level_shift_ratio", np.nan)
    if feat.get("recovery_phase", 0.0) > 0 or feat.get("normalization_phase", 0.0) > 0:
        base = max(base, 0.60)
    if pd.notna(shift) and np.isfinite(shift) and abs(float(shift) - 1.0) > 0.25:
        base = max(base, 0.65)
    return float(np.clip(base, 0.0, 0.90))


def recovery_anchor_prediction(feat: Dict[str, float], fallback_recent: float) -> float:
    baseline = feat.get("pre_covid_baseline_same_week", np.nan)
    progress = float(np.clip(feat.get("recovery_progress", 0.0), 0.0, 1.0))
    recent_candidates = [
        feat.get("target_ma_4w", np.nan),
        feat.get("target_lag_1w", np.nan),
        fallback_recent,
    ]
    recent = next((float(v) for v in recent_candidates if pd.notna(v) and np.isfinite(v)), np.nan)
    if pd.isna(baseline) or not np.isfinite(baseline):
        return max(recent, 0.0) if np.isfinite(recent) else 0.0
    if pd.isna(recent) or not np.isfinite(recent):
        return max(float(baseline), 0.0)
    return max((1.0 - progress) * recent + progress * float(baseline), 0.0)


def lv1_prediction_blend_weights(feat: Dict[str, float]) -> Tuple[float, float, float]:
    if feat.get("recovery_phase", 0.0) <= 0.0 and feat.get("normalization_phase", 0.0) <= 0.0:
        model_weight = dynamic_model_weight(feat)
        return model_weight, 1.0 - model_weight, 0.0

    progress = float(np.clip(feat.get("recovery_progress", 0.0), 0.0, 1.0))
    recovery_weight = 0.10 + 0.25 * progress
    model_weight = 0.65 - 0.15 * progress
    same_iso_weight = 1.0 - model_weight - recovery_weight
    weights = np.asarray([model_weight, same_iso_weight, recovery_weight], dtype=float)
    weights = np.clip(weights, 0.0, 1.0)
    total = float(weights.sum())
    if total <= EPS:
        return dynamic_model_weight(feat), 1.0 - dynamic_model_weight(feat), 0.0
    weights = weights / total
    return float(weights[0]), float(weights[1]), float(weights[2])


def lv1_training_sample_weight(weekly: pd.DataFrame) -> pd.Series:
    week_id = weekly["week_id"].astype(str)
    weight = pd.Series(1.0, index=weekly.index, dtype=float)
    weight = weight.mask(week_id.between(COVID_DROP_START_WEEK_ID, COVID_DROP_END_WEEK_ID), 0.45)
    weight = weight.mask(week_id.between(RECOVERY_START_WEEK_ID, RECOVERY_END_WEEK_ID), 1.30)
    weight = weight.mask(week_id.ge(NORMALIZATION_START_WEEK_ID), 1.50)
    return weight


def recovery_weekly_guardrail(pred: float, feat: Dict[str, float]) -> float:
    baseline = feat.get("pre_covid_baseline_same_week", np.nan)
    if (
        feat.get("normalization_phase", 0.0) > 0
        and pd.notna(baseline)
        and np.isfinite(baseline)
        and baseline > EPS
        and pred < baseline * 0.85
    ):
        return 0.70 * pred + 0.30 * float(baseline)
    if (
        feat.get("recovery_phase", 0.0) > 0
        and pd.notna(baseline)
        and np.isfinite(baseline)
        and baseline > EPS
        and pred < baseline * 0.85
    ):
        return 0.85 * pred + 0.15 * float(baseline)
    return pred


def fit_lv1_weekly_model(
    weekly: pd.DataFrame,
    target_col: str,
    train_end_week_id: str,
    train_start_week_id: Optional[str] = None,
) -> Tuple[WeeklyBaseLogModel, pd.Series, pd.DataFrame]:
    known_mask = training_week_mask(weekly, train_end_week_id, train_start_week_id)
    complete_mask = weekly.get("complete_target_week", True)
    train_mask = known_mask & complete_mask & weekly[target_col].notna()
    train_target_values, feature_values = lv1_training_target_and_feature_values(weekly, target_col, known_mask)
    known_end_pos = train_end_position(weekly, train_end_week_id)
    features = build_lv1_weekly_feature_frame(weekly, feature_values, known_end_pos)
    y = np.log1p(train_target_values)
    usable = train_mask & train_target_values.notna()
    usable = usable & features.notna().sum(axis=1).ge(10)
    sample_weight = lv1_training_sample_weight(weekly).loc[usable]
    model = WeeklyBaseLogModel(alpha=120.0).fit(features.loc[usable], y.loc[usable], sample_weight=sample_weight)
    return model, feature_values, features


def forecast_weekly_base_recursive(
    weekly: pd.DataFrame,
    target_col: str,
    train_end_week_id: str,
    forecast_week_ids: List[str],
    train_start_week_id: Optional[str] = None,
) -> pd.DataFrame:
    model, values, _ = fit_lv1_weekly_model(weekly, target_col, train_end_week_id, train_start_week_id)

    forecast_set = set(forecast_week_ids)
    if not forecast_set:
        return pd.DataFrame()
    max_forecast_week_id = max(forecast_set)
    predictions = []
    known_end_pos = train_end_position(weekly, train_end_week_id)
    values = values.copy()
    values.iloc[known_end_pos + 1 :] = np.nan
    exog_columns = lv1_exogenous_columns(weekly)
    exog_maps = make_exogenous_maps(weekly, exog_columns, known_end_pos)
    precovid_refs = make_precovid_reference_maps(weekly, values)
    train_mask = training_week_mask(weekly, train_end_week_id, train_start_week_id)
    same_iso_growth = estimate_same_iso_growth(weekly, values, train_end_week_id, train_start_week_id)
    fallback_median = float(values.loc[train_mask].median())

    for pos, row in weekly.iterrows():
        week_id = row["week_id"]
        if week_id <= train_end_week_id:
            continue
        if week_id > max_forecast_week_id:
            continue

        values_by_pos, values_by_iso = make_value_maps(weekly, values)
        feat_dict = row_lv1_weekly_features(
            weekly,
            int(pos),
            values_by_pos,
            values_by_iso,
            precovid_refs,
            exog_maps,
            exog_columns,
        )
        feat = pd.DataFrame([feat_dict]).reindex(columns=model.features)
        pred_log = float(model.predict(feat)[0])
        model_pred = max(float(np.expm1(pred_log)), 0.0)
        same_iso_pred = same_iso_reference_prediction(
            weekly,
            int(pos),
            values_by_pos,
            values_by_iso,
            same_iso_growth,
            fallback_median,
        )
        recovery_anchor = recovery_anchor_prediction(feat_dict, same_iso_pred)
        model_weight, same_iso_weight, recovery_weight = lv1_prediction_blend_weights(feat_dict)
        pred = (
            model_weight * model_pred
            + same_iso_weight * same_iso_pred
            + recovery_weight * recovery_anchor
        )
        pred = recovery_weekly_guardrail(pred, feat_dict)
        interval_log = float(np.log1p(pred))
        horizon = len(predictions) + 1
        pred_p10, pred_p90 = model.interval(interval_log, horizon, coverage=0.80)
        pred_p05, pred_p95 = model.interval(interval_log, horizon, coverage=0.90)
        values.iloc[int(pos)] = pred
        if week_id not in forecast_set:
            continue
        predictions.append(
            {
                "week_id": week_id,
                "week_start": row["week_start"],
                f"{target_col}_base": pred,
                f"{target_col}_base_p10": pred_p10,
                f"{target_col}_base_p90": pred_p90,
                f"{target_col}_base_p05": pred_p05,
                f"{target_col}_base_p95": pred_p95,
                f"{target_col}_lv1_model_pred": model_pred,
                f"{target_col}_same_iso_pred": same_iso_pred,
                f"{target_col}_lv1_model_weight": model_weight,
                f"{target_col}_same_iso_weight": same_iso_weight,
                f"{target_col}_recovery_weight": recovery_weight,
                f"{target_col}_recovery_anchor": recovery_anchor,
                f"{target_col}_lv1_backend": model.backend,
                f"{target_col}_pre_covid_baseline_same_week": feat_dict.get("pre_covid_baseline_same_week", np.nan),
                f"{target_col}_covid_adjusted_lag_52w": feat_dict.get("covid_adjusted_lag_52w", np.nan),
                f"{target_col}_recovery_safe_lag_52w": feat_dict.get("recovery_safe_lag_52w", np.nan),
                f"{target_col}_recovery_progress": feat_dict.get("recovery_progress", np.nan),
            }
        )

    return pd.DataFrame(predictions)


def historical_weekly_base(
    weekly: pd.DataFrame,
    target_col: str,
    train_end_week_id: str,
    train_start_week_id: Optional[str] = None,
) -> pd.DataFrame:
    train_mask = training_week_mask(weekly, train_end_week_id, train_start_week_id)
    mask = train_mask & weekly.get("complete_target_week", True) & weekly[target_col].notna()
    train_target_values, feature_values = lv1_training_target_and_feature_values(weekly, target_col, train_mask)
    precovid_refs = make_precovid_reference_maps(weekly, feature_values)
    out = weekly.loc[mask, ["week_id", "week_start"]].copy()
    out[f"{target_col}_base"] = train_target_values.loc[mask].clip(lower=0.0).to_numpy()
    out[f"{target_col}_pre_covid_baseline_same_week"] = [
        precovid_refs["baseline_by_week"].get(int(row.iso_week), precovid_refs["fallback_baseline"])
        for row in weekly.loc[mask, ["iso_week"]].itertuples(index=False)
    ]
    out[f"{target_col}_recovery_progress"] = [
        covid_regime_flags(str(row.week_id), pd.Timestamp(row.week_start))["recovery_progress"]
        for row in weekly.loc[mask, ["week_id", "week_start"]].itertuples(index=False)
    ]
    return out


def metric_frame(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {
            f"{prefix}_mae": np.nan,
            f"{prefix}_rmse": np.nan,
            f"{prefix}_wape": np.nan,
            f"{prefix}_r2": np.nan,
        }
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.sum(np.abs(y_true)))
    wape = float(np.sum(np.abs(err)) / denom) if denom > EPS else np.nan
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > EPS else np.nan
    return {
        f"{prefix}_mae": mae,
        f"{prefix}_rmse": rmse,
        f"{prefix}_wape": wape,
        f"{prefix}_r2": r2,
    }
