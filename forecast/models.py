from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .common import EPS, safe_div


class RidgeLogModel:
    def __init__(self, alpha: float = 25.0) -> None:
        self.alpha = alpha
        self.features: List[str] = []
        self.medians: Optional[pd.Series] = None
        self.means: Optional[pd.Series] = None
        self.stds: Optional[pd.Series] = None
        self.coef: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeLogModel":
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


class WeeklyLogModel:
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

    def fit(self, X: pd.DataFrame, y_log: pd.Series) -> "WeeklyLogModel":
        if len(X) < 20:
            raise ValueError(f"Need at least 20 training rows, got {len(X)}")
        self.features = list(X.columns)
        x = X.replace([np.inf, -np.inf], np.nan).astype(float)
        self.medians = x.median().fillna(0.0)
        x = x.fillna(self.medians)
        y = y_log.astype(float)

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
                self.model.fit(x, y)
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
                self.model.fit(x, y)
                self.backend = "xgboost"
                fitted = True
            except Exception:
                self.model = None

        if not fitted:
            self.ridge = RidgeLogModel(alpha=self.alpha).fit(x, y)
            self.backend = "ridge"

        raw = self._predict_raw(x)
        residual = y.to_numpy() - raw
        residual = residual[np.isfinite(residual)]
        if len(residual):
            # Mean residual corrects systematic log bias; half-variance corrects log-normal back-transform shrinkage.
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
        q = base_q * horizon_scale
        return max(float(np.expm1(pred_log - q)), 0.0), max(float(np.expm1(pred_log + q)), 0.0)


TARGET_LAGS = [1, 2, 4, 8, 13, 26, 52]
TARGET_WINDOWS = [4, 8, 13, 26]
EXOG_LAGS = [1, 2, 4, 8, 13]
EXOG_WINDOWS = [4, 13]
EXOGENOUS_BASE_COLUMNS = [
    "sessions",
    "unique_visitors",
    "page_views",
    "pageviews_per_session",
    "source_entropy",
    "orders_count",
    "orders_per_session",
    "stock_on_hand",
    "units_received",
    "stockout_sku_days",
    "active_sku_days",
    "days_of_supply",
    "fill_rate",
    "stockout_rate",
    "return_events",
    "returned_units",
    "refund_amount",
    "return_rate",
    "refund_rate",
    "review_count",
    "avg_rating",
    "low_rating_share",
    "new_customers",
    "promo_active_days",
    "promo_has_active",
    "promo_discount_sum",
    "stackable_promo_count",
    "discount_rate",
    "promo_intensity",
]
EXOG_GROWTH_COLUMNS = {
    "sessions",
    "page_views",
    "orders_count",
    "stockout_rate",
    "return_rate",
    "refund_rate",
    "avg_rating",
    "low_rating_share",
    "promo_active_days",
    "discount_rate",
}


MODEL_BACKEND_ALIASES = {
    "lgb": "lightgbm",
    "lgbm": "lightgbm",
    "xgb": "xgboost",
}
MODEL_BACKENDS = {"ridge", "lightgbm", "xgboost", "auto"}
DEFAULT_MODEL_BACKEND = "ridge"


def requested_model_backend() -> str:
    backend = os.environ.get("FORECAST_MODEL_BACKEND", DEFAULT_MODEL_BACKEND).strip().lower()
    backend = MODEL_BACKEND_ALIASES.get(backend, backend)
    return backend if backend in MODEL_BACKENDS else DEFAULT_MODEL_BACKEND

BASE_WEEKLY_FEATURES = [
    "time_index_week",
    "iso_week",
    "month",
    "quarter",
    "week_sin",
    "week_cos",
    "month_sin",
    "month_cos",
    "promo_active_days",
    "promo_has_active",
    "promo_discount_sum",
    "covid_period",
    "recovery_period",
    "post_covid_period",
    "weeks_since_covid_start",
    "weeks_since_recovery_start",
    "target_level_shift_ratio",
    "target_vol_4w",
    "target_growth_1w",
    "target_same_iso_week_1y",
    "target_yoy_hist",
]

WEEKLY_FEATURES = (
    BASE_WEEKLY_FEATURES
    + [f"target_lag_{lag}w" for lag in TARGET_LAGS]
    + [f"target_ma_{window}w" for window in TARGET_WINDOWS]
    + [f"{col}_lag_{lag}w" for col in EXOGENOUS_BASE_COLUMNS for lag in EXOG_LAGS]
    + [f"{col}_ma_{window}w" for col in EXOGENOUS_BASE_COLUMNS for window in EXOG_WINDOWS]
    + [f"{col}_growth_1w" for col in EXOGENOUS_BASE_COLUMNS if col in EXOG_GROWTH_COLUMNS]
)

MODEL_BLEND_WEIGHT = {
    "revenue_w": 0.55,
    "cogs_w": 0.40,
}


def make_value_maps(weekly: pd.DataFrame, values: pd.Series) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float]]:
    by_pos: Dict[int, float] = {}
    by_iso: Dict[Tuple[int, int], float] = {}
    for pos, row in weekly.iterrows():
        val = values.iloc[pos]
        if pd.notna(val):
            fval = float(val)
            by_pos[pos] = fval
            by_iso[(int(row["iso_year"]), int(row["iso_week"]))] = fval
    return by_pos, by_iso


def make_exogenous_maps(
    weekly: pd.DataFrame,
    columns: Iterable[str],
    known_end_pos: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    maps: Dict[str, Dict[str, Any]] = {}
    last_pos = len(weekly) - 1 if known_end_pos is None else int(known_end_pos)
    for col in columns:
        by_pos: Dict[int, float] = {}
        by_iso: Dict[Tuple[int, int], float] = {}
        by_week: Dict[int, List[float]] = {}
        if col in weekly.columns:
            for pos, row in weekly.iloc[: last_pos + 1].iterrows():
                val = row[col]
                if pd.notna(val) and np.isfinite(val):
                    fval = float(val)
                    by_pos[int(pos)] = fval
                    by_iso[(int(row["iso_year"]), int(row["iso_week"]))] = fval
                    by_week.setdefault(int(row["iso_week"]), []).append(fval)
        vals = list(by_pos.values())
        fallback = float(np.nanmedian(vals)) if vals else 0.0
        maps[col] = {
            "by_pos": by_pos,
            "by_iso": by_iso,
            "by_week": by_week,
            "fallback": fallback,
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
    iso_year = int(row["iso_year"])
    iso_week = int(row["iso_week"])
    same_last_year = by_iso.get((iso_year - 1, iso_week))
    if same_last_year is not None and np.isfinite(same_last_year):
        return float(same_last_year)

    same_week_history = by_week.get(iso_week, [])
    if same_week_history:
        return float(np.nanmedian(same_week_history[-5:]))

    recent = [val for pos, val in sorted(by_pos.items()) if pos < target_pos][-13:]
    if recent:
        return float(np.nanmean(recent))
    return fallback


def mean_finite(values: Iterable[float], fallback: float = np.nan) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else fallback


def std_finite(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanstd(arr)) if np.isfinite(arr).sum() >= 2 else np.nan


def regime_features(row: pd.Series) -> Dict[str, float]:
    week_start = pd.Timestamp(row["week_start"])
    covid_start = pd.Timestamp("2020-03-02")
    recovery_start = pd.Timestamp("2021-07-05")
    week_id = str(row["week_id"])
    return {
        "covid_period": float("2020-W10" <= week_id <= "2021-W26"),
        "recovery_period": float("2021-W27" <= week_id <= "2022-W52"),
        "post_covid_period": float(week_id >= "2023-W01"),
        "weeks_since_covid_start": max(float((week_start - covid_start).days) / 7.0, 0.0),
        "weeks_since_recovery_start": max(float((week_start - recovery_start).days) / 7.0, 0.0),
    }


def row_weekly_features(
    weekly: pd.DataFrame,
    pos: int,
    values_by_pos: Dict[int, float],
    values_by_iso: Dict[Tuple[int, int], float],
    exog_maps: Dict[str, Dict[str, Any]],
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
        "promo_active_days": float(row.get("promo_active_days", 0.0)),
        "promo_has_active": float(row.get("promo_has_active", 0.0)),
        "promo_discount_sum": float(row.get("promo_discount_sum", 0.0)),
    }
    features.update(regime_features(row))

    for lag in TARGET_LAGS:
        features[f"target_lag_{lag}w"] = values_by_pos.get(pos - lag, np.nan)

    for window in TARGET_WINDOWS:
        vals = [values_by_pos.get(pos - lag, np.nan) for lag in range(1, window + 1)]
        features[f"target_ma_{window}w"] = mean_finite(vals)
    vals4 = [values_by_pos.get(pos - lag, np.nan) for lag in range(1, 5)]
    features["target_vol_4w"] = std_finite(vals4)

    lag1 = features["target_lag_1w"]
    lag2 = features["target_lag_2w"]
    features["target_growth_1w"] = np.clip(safe_div(lag1 - lag2, lag2), -3.0, 3.0) if pd.notna(lag1) and pd.notna(lag2) else np.nan

    same_last_year = values_by_iso.get((int(row["iso_year"]) - 1, int(row["iso_week"])), np.nan)
    features["target_same_iso_week_1y"] = same_last_year
    features["target_yoy_hist"] = np.clip(safe_div(lag1 - same_last_year, same_last_year), -3.0, 3.0) if pd.notna(lag1) and pd.notna(same_last_year) else np.nan
    recent_13 = mean_finite([values_by_pos.get(pos - lag, np.nan) for lag in range(1, 14)])
    baseline_52 = mean_finite([values_by_pos.get(pos - lag, np.nan) for lag in range(1, 53)])
    features["target_level_shift_ratio"] = safe_div(recent_13, baseline_52) if pd.notna(recent_13) and pd.notna(baseline_52) else np.nan

    for col in EXOGENOUS_BASE_COLUMNS:
        for lag in EXOG_LAGS:
            features[f"{col}_lag_{lag}w"] = exogenous_value(weekly, pos - lag, col, exog_maps)
        for window in EXOG_WINDOWS:
            vals = [exogenous_value(weekly, pos - lag, col, exog_maps) for lag in range(1, window + 1)]
            features[f"{col}_ma_{window}w"] = mean_finite(vals, 0.0)
        if col in EXOG_GROWTH_COLUMNS:
            exog_lag1 = features[f"{col}_lag_1w"]
            exog_lag2 = features[f"{col}_lag_2w"]
            features[f"{col}_growth_1w"] = np.clip(safe_div(exog_lag1 - exog_lag2, exog_lag2), -3.0, 3.0)
    return features


def build_weekly_feature_frame(
    weekly: pd.DataFrame,
    target_col: str,
    values: pd.Series,
    known_end_pos: Optional[int] = None,
) -> pd.DataFrame:
    values_by_pos, values_by_iso = make_value_maps(weekly, values)
    exog_maps = make_exogenous_maps(weekly, EXOGENOUS_BASE_COLUMNS, known_end_pos=known_end_pos)
    rows = [row_weekly_features(weekly, pos, values_by_pos, values_by_iso, exog_maps) for pos in range(len(weekly))]
    return pd.DataFrame(rows, index=weekly.index)[WEEKLY_FEATURES]


def fit_weekly_model(weekly: pd.DataFrame, target_col: str, train_mask: pd.Series) -> Tuple[WeeklyLogModel, pd.Series]:
    values = weekly[target_col].copy()
    features = build_weekly_feature_frame(weekly, target_col, values)
    y = np.log1p(weekly[target_col])
    usable = train_mask & weekly[target_col].notna()
    usable = usable & features.notna().sum(axis=1).ge(24)
    model = WeeklyLogModel(alpha=80.0).fit(features.loc[usable], y.loc[usable])
    return model, features.loc[usable].median().fillna(0.0)


def estimate_same_iso_growth(weekly: pd.DataFrame, target_col: str, train_end_week_id: str) -> float:
    train = weekly[weekly["week_id"].le(train_end_week_id) & weekly[target_col].notna()].copy()
    by_iso = {
        (int(row["iso_year"]), int(row["iso_week"])): float(row[target_col])
        for _, row in train.iterrows()
    }
    ratios = []
    for _, row in train.iterrows():
        prev = by_iso.get((int(row["iso_year"]) - 1, int(row["iso_week"])))
        if prev is not None and prev > EPS:
            ratios.append(float(row[target_col]) / prev)
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


def train_end_position(weekly: pd.DataFrame, train_end_week_id: str) -> int:
    eligible = weekly.index[weekly["week_id"].le(train_end_week_id)]
    if len(eligible) == 0:
        return 0
    return int(eligible.max())


def dynamic_model_weight(target_col: str, feat: Dict[str, float]) -> float:
    base = MODEL_BLEND_WEIGHT.get(target_col, 0.40)
    shift = feat.get("target_level_shift_ratio", np.nan)
    if feat.get("post_covid_period", 0.0) > 0:
        base = max(base, 0.55)
    if pd.notna(shift) and np.isfinite(shift) and abs(float(shift) - 1.0) > 0.25:
        base = max(base, 0.50)
    return float(np.clip(base, 0.0, 0.90))


def forecast_weekly_recursive(
    weekly: pd.DataFrame,
    target_col: str,
    train_end_week_id: str,
    forecast_week_ids: List[str],
) -> pd.DataFrame:
    train_mask = weekly["week_id"].le(train_end_week_id) & weekly[target_col].notna()
    model, _ = fit_weekly_model(weekly, target_col, train_mask)

    values = weekly[target_col].copy()
    forecast_set = set(forecast_week_ids)
    predictions = []
    known_end_pos = train_end_position(weekly, train_end_week_id)
    exog_maps = make_exogenous_maps(weekly, EXOGENOUS_BASE_COLUMNS, known_end_pos=known_end_pos)
    same_iso_growth = estimate_same_iso_growth(weekly, target_col, train_end_week_id)
    fallback_median = float(weekly.loc[train_mask, target_col].median())

    for pos, row in weekly.iterrows():
        week_id = row["week_id"]
        if week_id <= train_end_week_id:
            continue
        if week_id not in forecast_set:
            continue

        values_by_pos, values_by_iso = make_value_maps(weekly, values)
        feat_dict = row_weekly_features(weekly, pos, values_by_pos, values_by_iso, exog_maps)
        feat = pd.DataFrame([feat_dict])[WEEKLY_FEATURES]
        pred_log = float(model.predict(feat)[0])
        model_pred = max(float(np.expm1(pred_log)), 0.0)
        same_iso_pred = same_iso_reference_prediction(
            weekly,
            pos,
            values_by_pos,
            values_by_iso,
            same_iso_growth,
            fallback_median,
        )
        model_weight = dynamic_model_weight(target_col, feat_dict)
        pred = model_weight * model_pred + (1.0 - model_weight) * same_iso_pred
        interval_log = float(np.log1p(pred))
        horizon = len(predictions) + 1
        pred_p10, pred_p90 = model.interval(interval_log, horizon, coverage=0.80)
        pred_p05, pred_p95 = model.interval(interval_log, horizon, coverage=0.90)
        values.iloc[pos] = pred
        predictions.append(
            {
                "week_id": week_id,
                "week_start": row["week_start"],
                f"{target_col}_pred": pred,
                f"{target_col}_pred_p10": pred_p10,
                f"{target_col}_pred_p90": pred_p90,
                f"{target_col}_pred_p05": pred_p05,
                f"{target_col}_pred_p95": pred_p95,
                f"{target_col}_model_pred": model_pred,
                f"{target_col}_same_iso_pred": same_iso_pred,
                f"{target_col}_model_weight": model_weight,
                f"{target_col}_model_backend": model.backend,
            }
        )

    return pd.DataFrame(predictions)


def forecast_weekly_cogs_ratio_recursive(
    weekly: pd.DataFrame,
    train_end_week_id: str,
    forecast_week_ids: List[str],
) -> pd.DataFrame:
    ratio = (weekly["cogs_w"] / weekly["revenue_w"]).replace([np.inf, -np.inf], np.nan)
    ratio = ratio.clip(lower=0.60, upper=1.20)
    values = ratio.copy()
    forecast_set = set(forecast_week_ids)
    train_mask = weekly["week_id"].le(train_end_week_id) & values.notna()
    fallback_median = float(values.loc[train_mask].median())
    predictions = []

    for pos, row in weekly.iterrows():
        week_id = row["week_id"]
        if week_id <= train_end_week_id:
            continue
        if week_id not in forecast_set:
            continue

        values_by_pos, values_by_iso = make_value_maps(weekly, values)
        same_last_year = values_by_iso.get((int(row["iso_year"]) - 1, int(row["iso_week"])))
        recent = [values_by_pos.get(pos - lag, np.nan) for lag in range(1, 14)]
        recent = [float(v) for v in recent if pd.notna(v) and np.isfinite(v)]
        recent_median = float(np.median(recent)) if recent else fallback_median

        if same_last_year is not None and np.isfinite(same_last_year):
            pred = 0.60 * float(same_last_year) + 0.40 * recent_median
        else:
            pred = recent_median
        pred = float(np.clip(pred, 0.65, 1.15))
        values.iloc[pos] = pred
        predictions.append(
            {
                "week_id": week_id,
                "week_start": row["week_start"],
                "cogs_ratio_pred": pred,
            }
        )

    return pd.DataFrame(predictions)


def forecast_cogs_from_revenue(
    weekly: pd.DataFrame,
    revenue_pred: pd.DataFrame,
    train_end_week_id: str,
    forecast_week_ids: List[str],
) -> pd.DataFrame:
    ratio_pred = forecast_weekly_cogs_ratio_recursive(weekly, train_end_week_id, forecast_week_ids)
    revenue_cols = ["week_id", "week_start", "revenue_w_pred"]
    revenue_cols += [c for c in ["revenue_w_pred_p10", "revenue_w_pred_p90", "revenue_w_pred_p05", "revenue_w_pred_p95"] if c in revenue_pred.columns]
    out = revenue_pred[revenue_cols].merge(
        ratio_pred, on=["week_id", "week_start"], how="left"
    )
    out["cogs_ratio_pred"] = out["cogs_ratio_pred"].fillna(out["cogs_ratio_pred"].median()).fillna(0.86)
    out["cogs_w_pred"] = (out["revenue_w_pred"] * out["cogs_ratio_pred"]).clip(lower=0.0)
    for src, dst in [
        ("revenue_w_pred_p10", "cogs_w_pred_p10"),
        ("revenue_w_pred_p90", "cogs_w_pred_p90"),
        ("revenue_w_pred_p05", "cogs_w_pred_p05"),
        ("revenue_w_pred_p95", "cogs_w_pred_p95"),
    ]:
        if src in out.columns:
            out[dst] = (out[src] * out["cogs_ratio_pred"]).clip(lower=0.0)
    keep_cols = ["week_id", "week_start", "cogs_w_pred", "cogs_ratio_pred"]
    keep_cols += [c for c in ["cogs_w_pred_p10", "cogs_w_pred_p90", "cogs_w_pred_p05", "cogs_w_pred_p95"] if c in out.columns]
    return out[keep_cols]


def metric_frame(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
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
