from __future__ import annotations

import os
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .common import EPS, Config, add_calendar_columns, read_csv, safe_div


def _slug(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_")
    return text or "unknown"


def aggregate_sales_daily(cfg: Config) -> pd.DataFrame:
    sales = read_csv(cfg.data_dir / "sales.csv", parse_dates=["Date"])
    out = sales.rename(columns={"Date": "date", "Revenue": "revenue", "COGS": "cogs"})
    out = out.groupby("date", as_index=False)[["revenue", "cogs"]].sum()

    net_mode = os.environ.get("FORECAST_NET_REVENUE", "0").strip().lower() in {"1", "true", "yes", "on"}
    if not net_mode:
        return out

    returns = read_csv(cfg.data_dir / "returns.csv", parse_dates=["return_date"])
    if not returns.empty:
        orders = read_csv(cfg.data_dir / "orders.csv", parse_dates=["order_date"], usecols=["order_id", "order_date"])
        returns = returns.merge(orders, on="order_id", how="left")
        refunds = (
            returns.groupby("order_date", as_index=False)["refund_amount"].sum()
            .rename(columns={"order_date": "date", "refund_amount": "refund_amount"})
        )
        out = out.merge(refunds, on="date", how="left")
        out["refund_amount"] = out["refund_amount"].fillna(0.0)
        # Net revenue by subtracting refunds mapped to original order_date.
        out["revenue"] = (out["revenue"] - out["refund_amount"]).clip(lower=0.0)
        out = out.drop(columns=["refund_amount"])

    return out


def aggregate_web_traffic_daily(cfg: Config) -> pd.DataFrame:
    wt = read_csv(cfg.data_dir / "web_traffic.csv", parse_dates=["date"])
    wt["bounce_x_sessions"] = wt["bounce_rate"] * wt["sessions"]
    wt["duration_x_sessions"] = wt["avg_session_duration_sec"] * wt["sessions"]
    out = wt.groupby("date", as_index=False).agg(
        sessions=("sessions", "sum"),
        unique_visitors=("unique_visitors", "sum"),
        page_views=("page_views", "sum"),
        bounce_x_sessions=("bounce_x_sessions", "sum"),
        duration_x_sessions=("duration_x_sessions", "sum"),
    )
    out["bounce_rate"] = safe_div(out["bounce_x_sessions"], out["sessions"])
    out["avg_session_duration_sec"] = safe_div(out["duration_x_sessions"], out["sessions"])
    out["pageviews_per_session"] = safe_div(out["page_views"], out["sessions"])
    return out.drop(columns=["bounce_x_sessions", "duration_x_sessions"])


def aggregate_orders_daily(cfg: Config) -> pd.DataFrame:
    orders = read_csv(cfg.data_dir / "orders.csv", parse_dates=["order_date"])
    out = orders.groupby("order_date", as_index=False).agg(
        orders_count=("order_id", "nunique"),
        active_customers=("customer_id", "nunique"),
    )
    status_counts = orders.pivot_table(
        index="order_date",
        columns="order_status",
        values="order_id",
        aggfunc="nunique",
        fill_value=0,
    ).reset_index()
    status_counts = status_counts.rename(columns={c: f"{_slug(c)}_orders" for c in status_counts.columns if c != "order_date"})
    out = out.merge(status_counts, on="order_date", how="left")
    payment_counts = orders.pivot_table(
        index="order_date",
        columns="payment_method",
        values="order_id",
        aggfunc="nunique",
        fill_value=0,
    ).reset_index()
    payment_counts = payment_counts.rename(
        columns={c: f"orders_payment_{_slug(c)}" for c in payment_counts.columns if c != "order_date"}
    )
    out = out.merge(payment_counts, on="order_date", how="left")
    out = out.rename(columns={"order_date": "date"})
    for col in [c for c in out.columns if c != "date"]:
        out[col] = out[col].fillna(0.0)
    out["cancel_rate"] = safe_div(out.get("cancelled_orders", 0.0), out["orders_count"]).fillna(0.0)
    out["cod_order_share"] = safe_div(out.get("orders_payment_cod", 0.0), out["orders_count"]).fillna(0.0)
    return out


def aggregate_basket_daily(cfg: Config) -> pd.DataFrame:
    orders = read_csv(cfg.data_dir / "orders.csv", parse_dates=["order_date"], usecols=["order_id", "order_date"])
    items = read_csv(cfg.data_dir / "order_items.csv")
    products = read_csv(cfg.data_dir / "products.csv", usecols=["product_id", "category"])
    items["gross_before_discount"] = items["quantity"] * items["unit_price"]
    items["net_before_refund"] = (items["gross_before_discount"] - items["discount_amount"]).clip(lower=0.0)
    items = items.merge(products, on="product_id", how="left")
    product_revenue = items.groupby("product_id", as_index=False)["net_before_refund"].sum()
    if product_revenue.empty:
        top_product_ids: set[object] = set()
    else:
        product_revenue = product_revenue.sort_values("net_before_refund", ascending=False).reset_index(drop=True)
        cumulative_share = product_revenue["net_before_refund"].cumsum() / max(product_revenue["net_before_refund"].sum(), EPS)
        top_count = max(1, int(np.ceil(len(product_revenue) * 0.20)))
        pareto_count = int(cumulative_share.le(0.80).sum()) + 1
        top_product_ids = set(product_revenue.head(max(top_count, pareto_count))["product_id"])
    items["promo_line"] = items["promo_id"].notna() | items["promo_id_2"].notna()
    items["top_product_line"] = items["product_id"].isin(top_product_ids)
    items["top_product_revenue"] = np.where(items["top_product_line"], items["net_before_refund"], 0.0)
    items["streetwear_revenue"] = np.where(items["category"].eq("Streetwear"), items["net_before_refund"], 0.0)
    items["promo_line_revenue"] = np.where(items["promo_line"], items["net_before_refund"], 0.0)
    items["promo_line_discount_amount"] = np.where(items["promo_line"], items["discount_amount"], 0.0)
    merged = items.merge(orders, on="order_id", how="left")
    out = merged.groupby("order_date", as_index=False).agg(
        units=("quantity", "sum"),
        gross_before_discount=("gross_before_discount", "sum"),
        discount_amount=("discount_amount", "sum"),
        net_item_revenue_before_refund=("net_before_refund", "sum"),
        top_product_revenue=("top_product_revenue", "sum"),
        streetwear_revenue=("streetwear_revenue", "sum"),
        promo_line_revenue=("promo_line_revenue", "sum"),
        promo_line_discount_amount=("promo_line_discount_amount", "sum"),
        promo_orders=("order_id", lambda s: s[merged.loc[s.index, "promo_line"]].nunique()),
    )
    order_counts = merged.groupby("order_date")["order_id"].nunique().rename("line_order_count")
    out = out.merge(order_counts, on="order_date", how="left")
    out = out.rename(columns={"order_date": "date"})
    out["top_product_revenue_share"] = safe_div(out["top_product_revenue"], out["net_item_revenue_before_refund"]).fillna(0.0)
    out["streetwear_revenue_share"] = safe_div(out["streetwear_revenue"], out["net_item_revenue_before_refund"]).fillna(0.0)
    out["promo_order_share"] = safe_div(out["promo_orders"], out["line_order_count"]).fillna(0.0)
    out["promo_discount_rate"] = safe_div(
        out["promo_line_discount_amount"],
        out["promo_line_revenue"] + out["promo_line_discount_amount"],
    ).fillna(0.0)
    return out.drop(columns=["line_order_count"])


def aggregate_inventory_daily(cfg: Config) -> pd.DataFrame:
    inv = read_csv(cfg.data_dir / "inventory.csv", parse_dates=["snapshot_date"])
    out = inv.groupby("snapshot_date", as_index=False).agg(
        stock_on_hand=("stock_on_hand", "sum"),
        units_received=("units_received", "sum"),
        inventory_units_sold=("units_sold", "sum"),
        stockout_days=("stockout_days", "sum"),
        days_of_supply=("days_of_supply", "mean"),
        fill_rate=("fill_rate", "mean"),
        stockout_sku_days=("stockout_flag", "sum"),
        active_sku_days=("product_id", "nunique"),
        reorder_sku_days=("reorder_flag", "sum"),
        sell_through_rate=("sell_through_rate", "mean"),
    )
    return out.rename(columns={"snapshot_date": "date"})


def aggregate_promotions_daily(cfg: Config, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    promos = read_csv(cfg.data_dir / "promotions.csv", parse_dates=["start_date", "end_date"])
    dates = pd.DataFrame({"date": pd.date_range(start_dt, end_dt, freq="D")})
    promo_defaults = {
        "active_promo_count": 0,
        "avg_promo_discount_value": 0.0,
        "stackable_promo_count": 0,
        "promo_start_day": 0,
        "promo_end_day": 0,
        "days_to_promo": 999.0,
        "days_since_promo_start": 999.0,
        "days_until_promo_end": 999.0,
        "promo_seg_code": 0.0,
        "promo_discount_value": 0.0,
        "promo_type_fixed": 0.0,
        "promo_impact_score": 0.0,
        "promo_projected_flag": 0.0,
        "active_promo_count_actual": 0.0,
        "avg_promo_discount_value_actual": 0.0,
    }

    def nanmin_or_cap(values: np.ndarray, cap: float = 999.0) -> np.ndarray:
        finite = np.isfinite(values)
        result = np.full(values.shape[0], cap, dtype=float)
        if finite.any():
            row_has_value = finite.any(axis=1)
            safe_values = np.where(finite, values, np.nan)
            result[row_has_value] = np.nanmin(safe_values[row_has_value], axis=1)
        return np.minimum(result, cap)
    if promos.empty:
        for col, value in promo_defaults.items():
            dates[col] = value
        return dates

    def attach_projected_promos(merged: pd.DataFrame, source_promos: pd.DataFrame) -> pd.DataFrame:
        project_promos = os.environ.get("FORECAST_PROMO_PROJECTION", "1").strip().lower() in {"1", "true", "yes", "on"}
        if not project_promos:
            return merged
        if source_promos.empty:
            return merged

        def map_date_to_year(day: pd.Timestamp, target_year: int) -> pd.Timestamp:
            try:
                return pd.Timestamp(year=int(target_year), month=int(day.month), day=int(day.day))
            except ValueError:
                return pd.Timestamp(year=int(target_year), month=2, day=28)

        def promo_seg_code(category_value: object, discount: float) -> float:
            category_text = "" if pd.isna(category_value) else str(category_value)
            if "Outdoor" in category_text:
                return 4.0
            if "Streetwear" in category_text:
                return 3.0
            if discount >= 18.0:
                return 2.0
            return 1.0

        active_years: set[int] = set()
        for promo in source_promos.itertuples(index=False):
            start = pd.Timestamp(promo.start_date).normalize()
            end = pd.Timestamp(promo.end_date).normalize()
            if pd.isna(start) or pd.isna(end):
                continue
            active_years.update(range(int(start.year), int(end.year) + 1))
        if not active_years:
            return merged

        def reference_year_for(target_year: int) -> int:
            preferred = int(target_year) - 2
            if preferred in active_years:
                return preferred
            same_parity = [year for year in active_years if year < target_year and year % 2 == target_year % 2]
            if same_parity:
                return max(same_parity)
            previous = [year for year in active_years if year < target_year]
            return max(previous) if previous else max(active_years)

        out = merged.copy()
        last_known_promo_end = source_promos["end_date"].max()
        future_dates = out.loc[out["date"].gt(last_known_promo_end), "date"]
        target_years = sorted(future_dates.dt.year.dropna().astype(int).unique().tolist())
        projected_rows = []
        horizon_start = pd.Timestamp(out["date"].min()).normalize()
        horizon_end = pd.Timestamp(out["date"].max()).normalize()

        for target_year in target_years:
            ref_year = reference_year_for(target_year)
            ref_start = pd.Timestamp(year=ref_year, month=1, day=1)
            ref_end = pd.Timestamp(year=ref_year, month=12, day=31)
            ref_promos = source_promos[
                source_promos["start_date"].le(ref_end) & source_promos["end_date"].ge(ref_start)
            ].copy()
            for promo in ref_promos.itertuples(index=False):
                promo_start = max(pd.Timestamp(promo.start_date).normalize(), ref_start)
                promo_end = min(pd.Timestamp(promo.end_date).normalize(), ref_end)
                target_start = map_date_to_year(promo_start, target_year)
                target_end = map_date_to_year(promo_end, target_year)
                if target_end < target_start:
                    continue
                target_start = max(target_start, horizon_start, pd.Timestamp(last_known_promo_end) + pd.Timedelta(days=1))
                target_end = min(target_end, horizon_end)
                if target_end < target_start:
                    continue
                discount = float(getattr(promo, "discount_value", 0.0) or 0.0)
                seg_code = promo_seg_code(getattr(promo, "applicable_category", ""), discount)
                promo_type_value = getattr(promo, "promo_type", "")
                promo_type_text = "" if pd.isna(promo_type_value) else str(promo_type_value).strip().lower()
                for dt in pd.date_range(target_start, target_end, freq="D"):
                    projected_rows.append(
                        {
                            "date": dt,
                            "promo_discount_value_projected": discount,
                            "promo_seg_code_projected": seg_code,
                            "promo_type_fixed_projected": float(promo_type_text == "fixed"),
                            "stackable_promo_count_projected": float(getattr(promo, "stackable_flag", 0.0) or 0.0),
                        }
                    )
        if not projected_rows:
            return out

        projected_daily = pd.DataFrame(projected_rows)
        projected_daily["promo_impact_score_projected"] = (
            projected_daily["promo_seg_code_projected"] * projected_daily["promo_discount_value_projected"]
        )
        stackable = (
            projected_daily.groupby("date", as_index=False)["stackable_promo_count_projected"]
            .max()
        )
        projected_daily = projected_daily.sort_values(
            ["date", "promo_discount_value_projected", "promo_impact_score_projected"],
            ascending=[True, False, False],
        ).drop_duplicates("date", keep="first")
        projected_daily = projected_daily.drop(columns=["stackable_promo_count_projected"]).merge(stackable, on="date", how="left")
        projected_daily = projected_daily.sort_values("date").reset_index(drop=True)
        previous_date = projected_daily["date"].shift(1)
        next_date = projected_daily["date"].shift(-1)
        projected_daily["promo_start_day_projected"] = (
            previous_date.isna() | projected_daily["date"].sub(previous_date).dt.days.ne(1)
        ).astype(float)
        projected_daily["promo_end_day_projected"] = (
            next_date.isna() | next_date.sub(projected_daily["date"]).dt.days.ne(1)
        ).astype(float)
        projected_daily["_projected_block"] = projected_daily["promo_start_day_projected"].cumsum()
        block_start = projected_daily.groupby("_projected_block")["date"].transform("min")
        block_end = projected_daily.groupby("_projected_block")["date"].transform("max")
        projected_daily["days_since_promo_start_projected"] = projected_daily["date"].sub(block_start).dt.days.astype(float)
        projected_daily["days_until_promo_end_projected"] = block_end.sub(projected_daily["date"]).dt.days.astype(float)
        projected_daily["active_promo_count_projected"] = 1.0

        out = out.merge(projected_daily.drop(columns=["_projected_block"]), on="date", how="left")
        projected = (
            out["date"].gt(last_known_promo_end)
            & out["active_promo_count"].fillna(0).le(0)
            & out["active_promo_count_projected"].fillna(0).gt(0)
        )
        out.loc[projected, "active_promo_count"] = out.loc[projected, "active_promo_count_projected"]
        out.loc[projected, "avg_promo_discount_value"] = out.loc[projected, "promo_discount_value_projected"]
        out.loc[projected, "stackable_promo_count"] = out.loc[projected, "stackable_promo_count_projected"]
        out.loc[projected, "promo_start_day"] = out.loc[projected, "promo_start_day_projected"]
        out.loc[projected, "promo_end_day"] = out.loc[projected, "promo_end_day_projected"]
        out.loc[projected, "days_since_promo_start"] = out.loc[projected, "days_since_promo_start_projected"]
        out.loc[projected, "days_until_promo_end"] = out.loc[projected, "days_until_promo_end_projected"]
        out.loc[projected, "promo_seg_code"] = out.loc[projected, "promo_seg_code_projected"]
        out.loc[projected, "promo_discount_value"] = out.loc[projected, "promo_discount_value_projected"]
        out.loc[projected, "promo_type_fixed"] = out.loc[projected, "promo_type_fixed_projected"]
        out.loc[projected, "promo_impact_score"] = out.loc[projected, "promo_impact_score_projected"]
        out.loc[projected, "promo_projected_flag"] = 1.0
        return out.drop(
            columns=[
                "active_promo_count_projected",
                "promo_seg_code_projected",
                "promo_discount_value_projected",
                "promo_type_fixed_projected",
                "promo_impact_score_projected",
                "stackable_promo_count_projected",
                "promo_start_day_projected",
                "promo_end_day_projected",
                "days_since_promo_start_projected",
                "days_until_promo_end_projected",
            ],
            errors="ignore",
        )

    promos = promos.dropna(subset=["start_date", "end_date"]).copy()
    promos["start_date"] = promos["start_date"].dt.normalize()
    promos["end_date"] = promos["end_date"].dt.normalize()
    known_end_raw = os.environ.get("FORECAST_PROMO_KNOWN_END_DATE", "").strip()
    if known_end_raw:
        known_end = pd.Timestamp(known_end_raw).normalize()
        promos = promos[promos["end_date"].le(known_end)].copy()
        if promos.empty:
            for col, value in promo_defaults.items():
                dates[col] = value
            return dates
    promos = promos[promos["end_date"].ge(start_dt) & promos["start_date"].le(end_dt)].copy()
    if promos.empty:
        for col, value in promo_defaults.items():
            dates[col] = value
        return dates

    expanded = dates.assign(_key=1).merge(
        promos[["promo_id", "start_date", "end_date", "discount_value", "stackable_flag", "promo_type", "applicable_category"]].assign(_key=1),
        on="_key",
        how="inner",
    ).drop(columns="_key")
    expanded = expanded[expanded["date"].between(expanded["start_date"], expanded["end_date"])].copy()
    expanded["promo_start_day"] = expanded["date"].eq(expanded["start_date"]).astype(int)
    expanded["promo_end_day"] = expanded["date"].eq(expanded["end_date"]).astype(int)
    out = expanded.groupby("date", as_index=False).agg(
        active_promo_count=("promo_id", "nunique"),
        avg_promo_discount_value=("discount_value", "mean"),
        stackable_promo_count=("stackable_flag", "sum"),
        promo_start_day=("promo_start_day", "max"),
        promo_end_day=("promo_end_day", "max"),
    )
    expanded["promo_discount_value"] = expanded["discount_value"].astype(float)
    expanded["promo_type_fixed"] = expanded["promo_type"].eq("fixed").astype(float)
    category = expanded["applicable_category"].fillna("").astype(str)
    expanded["promo_seg_code"] = 1.0
    expanded.loc[expanded["promo_discount_value"].ge(18.0), "promo_seg_code"] = 2.0
    expanded.loc[category.str.contains("Streetwear", na=False), "promo_seg_code"] = 3.0
    expanded.loc[category.str.contains("Outdoor", na=False), "promo_seg_code"] = 4.0
    strongest = expanded.sort_values(["date", "promo_discount_value"], ascending=[True, False])
    strongest = strongest.drop_duplicates("date", keep="first")[
        ["date", "promo_seg_code", "promo_discount_value", "promo_type_fixed"]
    ].copy()
    strongest["promo_impact_score"] = strongest["promo_seg_code"] * strongest["promo_discount_value"]
    promo_start_dates = promos["start_date"].dropna().dt.normalize().drop_duplicates().sort_values().to_numpy()
    promo_end_dates = promos["end_date"].dropna().dt.normalize().drop_duplicates().sort_values().to_numpy()
    if len(promo_start_dates):
        day_values = dates["date"].dt.normalize().to_numpy()
        signed_start_distances = (day_values[:, None] - promo_start_dates[None, :]).astype("timedelta64[D]").astype(float)
        future_start = np.where(signed_start_distances <= 0, np.abs(signed_start_distances), np.nan)
        past_start = np.where(signed_start_distances >= 0, signed_start_distances, np.nan)
        dates["days_to_promo"] = nanmin_or_cap(future_start)
        dates["days_since_promo_start"] = nanmin_or_cap(past_start)
    else:
        dates["days_to_promo"] = 999.0
        dates["days_since_promo_start"] = 999.0
    if len(promo_end_dates):
        day_values = dates["date"].dt.normalize().to_numpy()
        signed_end_distances = (promo_end_dates[None, :] - day_values[:, None]).astype("timedelta64[D]").astype(float)
        future_end = np.where(signed_end_distances >= 0, signed_end_distances, np.nan)
        dates["days_until_promo_end"] = nanmin_or_cap(future_end)
    else:
        dates["days_until_promo_end"] = 999.0
    merged = dates.merge(out, on="date", how="left")
    merged = merged.merge(strongest, on="date", how="left")
    for col, value in promo_defaults.items():
        if col in merged.columns:
            merged[col] = merged[col].fillna(value)
        else:
            merged[col] = value
    merged = attach_projected_promos(merged, promos)
    active = merged["active_promo_count"].fillna(0).gt(0)
    merged.loc[active, "days_to_promo"] = 0.0
    disable_lookahead = os.environ.get("FORECAST_DISABLE_PROMO_LOOKAHEAD", "1").strip().lower() in {"1", "true", "yes", "on"}
    if disable_lookahead:
        merged["days_to_promo"] = np.where(active, 0.0, 999.0)
        merged["days_until_promo_end"] = np.where(active, merged["days_until_promo_end"], 999.0)
    projected = merged["promo_projected_flag"].fillna(0).gt(0)
    merged["active_promo_count_actual"] = np.where(projected, 0.0, merged["active_promo_count"].fillna(0.0))
    merged["avg_promo_discount_value_actual"] = np.where(projected, 0.0, merged["avg_promo_discount_value"].fillna(0.0))
    return merged


def merge_daily(spine: pd.DataFrame, marts: Iterable[pd.DataFrame]) -> pd.DataFrame:
    out = spine.copy()
    for mart in marts:
        if mart["date"].duplicated().any():
            dupes = mart.loc[mart["date"].duplicated(), "date"].head().tolist()
            raise ValueError(f"Daily mart has duplicate dates before join: {dupes}")
        out = out.merge(mart, on="date", how="left")
    return out


def _expanding_calendar_profile(
    df: pd.DataFrame,
    col: str,
    group_cols: list[str],
    min_periods: int = 3,
) -> pd.Series:
    values = df[col].replace([np.inf, -np.inf], np.nan).astype(float)

    def prior_group_median(s: pd.Series) -> pd.Series:
        return s.shift(1).expanding(min_periods=min_periods).median()

    prof = values.groupby([df[g] for g in group_cols], group_keys=False).apply(prior_group_median)
    return prof.reindex(df.index)


def _prior_year_calendar_profile(df: pd.DataFrame, col: str) -> pd.Series:
    values = df[col].replace([np.inf, -np.inf], np.nan).astype(float)
    years = df["iso_year"].astype(int)
    month = df["month"].astype(int)
    iso_week = df["iso_week"].astype(int)
    iso_weekday = df["iso_weekday"].astype(int)
    fallback = float(values.dropna().median()) if values.notna().any() else 0.0
    out = []
    for idx in df.index:
        prior = years < int(years.loc[idx])
        candidates = values[prior & month.eq(int(month.loc[idx])) & iso_weekday.eq(int(iso_weekday.loc[idx]))].dropna()
        if candidates.empty:
            candidates = values[prior & iso_week.eq(int(iso_week.loc[idx])) & iso_weekday.eq(int(iso_weekday.loc[idx]))].dropna()
        if candidates.empty:
            candidates = values[prior & month.eq(int(month.loc[idx]))].dropna()
        if candidates.empty:
            candidates = values[prior].dropna()
        out.append(float(candidates.median()) if not candidates.empty else fallback)
    return pd.Series(out, index=df.index, dtype=float)


def add_forecast_safe_operational_profiles(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.sort_values("date").reset_index(drop=True).copy()
    out["orders_per_1000_sessions"] = safe_div(out.get("orders_count", np.nan), out.get("sessions", np.nan)) * 1000.0
    out["orders_per_session"] = safe_div(out.get("orders_count", np.nan), out.get("sessions", np.nan))
    out["revenue_per_session"] = safe_div(out.get("revenue", np.nan), out.get("sessions", np.nan))
    out["stockout_rate"] = safe_div(out.get("stockout_sku_days", np.nan), out.get("active_sku_days", np.nan))
    out["stock_pressure_index"] = out["stockout_rate"].fillna(0.0) * out.get("sessions", pd.Series(np.nan, index=out.index))
    out["promo_margin_pressure_actual"] = out.get("promo_order_share", 0.0).fillna(0.0) * out.get("promo_discount_rate", 0.0).fillna(0.0)
    out["orders_per_session_lag_14d"] = out["orders_per_session"].shift(14)
    out["revenue_per_session_lag_28d"] = out["revenue_per_session"].shift(28)
    out["funnel_efficiency_lag_28d"] = safe_div(
        out["revenue"].shift(1).rolling(28, min_periods=7).sum(),
        out.get("sessions", pd.Series(np.nan, index=out.index)).shift(1).rolling(28, min_periods=7).sum(),
    )
    out["streetwear_concentration_risk"] = safe_div(
        out.get("streetwear_revenue", pd.Series(np.nan, index=out.index)).shift(1).rolling(28, min_periods=7).sum(),
        out.get("net_item_revenue_before_refund", pd.Series(np.nan, index=out.index)).shift(1).rolling(28, min_periods=7).sum(),
    )

    profile_cols = [
        "orders_per_1000_sessions",
        "cancel_rate",
        "cod_order_share",
        "stockout_rate",
        "fill_rate",
        "stock_pressure_index",
        "top_product_revenue_share",
        "promo_order_share",
        "promo_discount_rate",
        "promo_margin_pressure_actual",
        "streetwear_concentration_risk",
    ]
    global_profiles: dict[str, pd.Series] = {}
    for col in profile_cols:
        if col not in out.columns:
            out[col] = np.nan
        values = out[col].replace([np.inf, -np.inf], np.nan).astype(float)
        expected = _prior_year_calendar_profile(out, col)
        global_profile = values.groupby(out["iso_year"]).transform("median").shift(366).ffill()
        fallback = float(values.dropna().median()) if values.notna().any() else 0.0
        expected = expected.replace([np.inf, -np.inf], np.nan).fillna(fallback)
        out[f"expected_{col}"] = expected.astype(float)
        global_profiles[col] = global_profile.fillna(fallback)

    out["expected_lost_sales_index"] = (
        out["expected_stockout_rate"].fillna(0.0).clip(lower=0.0, upper=1.0)
        * np.log1p(out["expected_orders_per_1000_sessions"].fillna(0.0).clip(lower=0.0))
    )
    out["promo_margin_pressure"] = (
        out.get("promo_flag", 0.0).fillna(0.0) * np.log1p(out.get("avg_promo_discount_value", 0.0).fillna(0.0))
        + out["expected_promo_margin_pressure_actual"].fillna(0.0)
    )
    payday = out.get("is_payday_window", pd.Series(0.0, index=out.index)).fillna(0.0).astype(float)
    month_start = out.get("is_month_start", pd.Series(0.0, index=out.index)).fillna(0.0).astype(float)
    month_end = out.get("is_month_end", pd.Series(0.0, index=out.index)).fillna(0.0).astype(float)
    promo_flag = out.get("promo_flag", pd.Series(0.0, index=out.index)).fillna(0.0).astype(float)
    expected_conversion = out["expected_orders_per_1000_sessions"].fillna(0.0).clip(lower=0.0)
    expected_top_mix = out["expected_top_product_revenue_share"].fillna(0.0).clip(lower=0.0, upper=1.0)
    out["is_payday_month_start"] = payday * month_start
    out["is_payday_month_end"] = payday * month_end
    out["is_payday_month_edge"] = payday * np.maximum(month_start, month_end)
    out["is_payday_promo"] = payday * promo_flag
    out["payday_expected_conversion"] = payday * np.log1p(expected_conversion)
    out["payday_expected_top_mix"] = payday * expected_top_mix
    return out


def build_daily_mart(cfg: Config, sample: pd.DataFrame) -> pd.DataFrame:
    sales_daily = aggregate_sales_daily(cfg)
    start_dt = sales_daily["date"].min()
    end_dt = pd.Timestamp(sample["Date"].max()).normalize()
    end_dt = end_dt + pd.Timedelta(days=6 - end_dt.weekday())
    spine = pd.DataFrame({"date": pd.date_range(start_dt, end_dt, freq="D")})

    marts = [
        sales_daily,
        aggregate_web_traffic_daily(cfg),
        aggregate_orders_daily(cfg),
        aggregate_basket_daily(cfg),
        aggregate_inventory_daily(cfg),
        aggregate_promotions_daily(cfg, start_dt, end_dt),
    ]
    daily = merge_daily(spine, marts).sort_values("date").reset_index(drop=True)
    daily = add_calendar_columns(daily, "date")
    daily["is_operational_crisis"] = daily["date"].dt.year.ge(2019).astype(float)
    daily["internal_stress_regime"] = daily["is_operational_crisis"]
    daily["time_index_day"] = np.arange(len(daily), dtype=float)
    daily["promo_flag"] = daily.get("active_promo_count", 0).fillna(0).gt(0).astype(float)
    daily = add_forecast_safe_operational_profiles(daily)

    zero_fill_prefixes = (
        "orders_payment_",
    )
    zero_fill_exact = {
        "sessions",
        "unique_visitors",
        "page_views",
        "orders_count",
        "active_customers",
        "created_orders",
        "paid_orders",
        "shipped_orders",
        "delivered_orders",
        "returned_orders",
        "cancelled_orders",
        "units",
        "gross_before_discount",
        "discount_amount",
        "net_item_revenue_before_refund",
        "top_product_revenue",
        "streetwear_revenue",
        "promo_line_revenue",
        "promo_line_discount_amount",
        "promo_orders",
        "stock_on_hand",
        "units_received",
        "inventory_units_sold",
        "stockout_days",
        "stockout_sku_days",
        "active_sku_days",
        "reorder_sku_days",
        "active_promo_count",
        "stackable_promo_count",
        "promo_start_day",
        "promo_end_day",
        "promo_projected_flag",
        "active_promo_count_actual",
    }
    for col in daily.columns:
        if col in zero_fill_exact or any(col.startswith(prefix) for prefix in zero_fill_prefixes):
            daily[col] = daily[col].fillna(0)

    rate_cols = [
        "bounce_rate",
        "avg_session_duration_sec",
        "pageviews_per_session",
        "orders_per_1000_sessions",
        "orders_per_session",
        "revenue_per_session",
        "orders_per_session_lag_14d",
        "revenue_per_session_lag_28d",
        "funnel_efficiency_lag_28d",
        "cancel_rate",
        "cod_order_share",
        "stockout_rate",
        "stock_pressure_index",
        "days_of_supply",
        "fill_rate",
        "sell_through_rate",
        "top_product_revenue_share",
        "streetwear_revenue_share",
        "streetwear_concentration_risk",
        "promo_order_share",
        "promo_discount_rate",
        "promo_margin_pressure_actual",
        "avg_promo_discount_value",
        "avg_promo_discount_value_actual",
        "days_to_promo",
        "days_since_promo_start",
        "days_until_promo_end",
    ]
    for col in rate_cols:
        if col in daily.columns:
            daily[col] = daily[col].fillna(0)

    daily = daily.copy()
    daily["promo_flag"] = daily.get("active_promo_count", 0).fillna(0).gt(0).astype(float)

    for target in ["revenue", "cogs"]:
        for lag in [1, 7, 14, 28]:
            daily[f"{target}_lag_{lag}d"] = daily[target].shift(lag)
        daily[f"{target}_ma_7d"] = daily[target].shift(1).rolling(7, min_periods=2).mean()
        daily[f"{target}_ma_28d"] = daily[target].shift(1).rolling(28, min_periods=7).mean()
        daily[f"{target}_shock_7d"] = (
            safe_div((daily[f"{target}_lag_1d"] - daily[f"{target}_ma_7d"]).abs(), daily[f"{target}_ma_7d"])
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(lower=0.0, upper=5.0)
        )

    if daily["date"].duplicated().any():
        raise ValueError("daily_mart has duplicate dates")

    return daily.sort_values("date").reset_index(drop=True)


def build_weekly_mart(daily: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in daily.columns if c not in {"date", "week_start", "week_id"} and pd.api.types.is_numeric_dtype(daily[c])]
    mean_cols = {
        "avg_promo_discount_value",
        "bounce_rate",
        "avg_session_duration_sec",
        "pageviews_per_session",
        "orders_per_1000_sessions",
        "cancel_rate",
        "cod_order_share",
        "stockout_rate",
        "stock_pressure_index",
        "days_of_supply",
        "fill_rate",
        "sell_through_rate",
        "top_product_revenue_share",
        "streetwear_revenue_share",
        "streetwear_concentration_risk",
        "promo_order_share",
        "promo_discount_rate",
        "promo_margin_pressure_actual",
        "expected_orders_per_1000_sessions",
        "expected_cancel_rate",
        "expected_cod_order_share",
        "expected_stockout_rate",
        "expected_stock_pressure_index",
        "expected_fill_rate",
        "expected_top_product_revenue_share",
        "expected_streetwear_concentration_risk",
        "expected_promo_order_share",
        "expected_promo_discount_rate",
        "expected_promo_margin_pressure_actual",
        "expected_lost_sales_index",
        "promo_margin_pressure",
        "orders_per_session_lag_14d",
        "revenue_per_session_lag_28d",
        "funnel_efficiency_lag_28d",
        "is_operational_crisis",
        "internal_stress_regime",
        "payday_expected_conversion",
        "payday_expected_top_mix",
        "days_to_promo",
        "days_since_promo_start",
        "days_until_promo_end",
    }
    excluded_calendar = {
        "iso_year",
        "iso_week",
        "iso_weekday",
        "month",
        "quarter",
        "day_of_month",
        "day_of_year",
        "is_weekend",
        "week_position_in_month",
    }
    def is_mean_col(col: str) -> bool:
        return col in mean_cols
    agg = {
        col: ("mean" if is_mean_col(col) else "sum")
        for col in numeric_cols
        if col not in excluded_calendar
    }
    grouped = daily.groupby("week_start")
    weekly = grouped.agg(agg).reset_index()

    first = grouped.first(numeric_only=False).reset_index()[
        ["week_start", "iso_year", "iso_week", "week_id", "month", "quarter"]
    ]
    weekly = weekly.drop(columns=[c for c in ["iso_year", "iso_week", "month", "quarter"] if c in weekly.columns], errors="ignore")
    weekly = first.merge(weekly, on="week_start", how="left")

    counts = grouped.agg(
        days_in_spine=("date", "count"),
        revenue_days=("revenue", lambda s: s.notna().sum()),
        cogs_days=("cogs", lambda s: s.notna().sum()),
    ).reset_index()
    weekly = weekly.merge(counts, on="week_start", how="left")

    lv1_source = daily.copy()
    holiday_flag = lv1_source["is_holiday"] if "is_holiday" in lv1_source.columns else pd.Series(0, index=lv1_source.index)
    lv1_source["_lv1_non_holiday_day"] = (
        lv1_source["revenue"].notna()
        & lv1_source["cogs"].notna()
        & holiday_flag.eq(0)
    )
    lv1_source["_revenue_non_holiday"] = lv1_source["revenue"].where(lv1_source["_lv1_non_holiday_day"])
    lv1_source["_cogs_non_holiday"] = lv1_source["cogs"].where(lv1_source["_lv1_non_holiday_day"])
    lv1_targets = lv1_source.groupby("week_start").agg(
        lv1_non_holiday_days=("_lv1_non_holiday_day", "sum"),
        revenue_w_non_holiday=("_revenue_non_holiday", "sum"),
        cogs_w_non_holiday=("_cogs_non_holiday", "sum"),
        holiday_days=("is_holiday", "sum"),
    ).reset_index()
    weekly = weekly.merge(lv1_targets, on="week_start", how="left")
    weekly["complete_spine_week"] = weekly["days_in_spine"].eq(7)
    weekly["complete_target_week"] = weekly["revenue_days"].eq(7) & weekly["cogs_days"].eq(7)

    weekly = weekly.rename(columns={"revenue": "revenue_w", "cogs": "cogs_w"})
    weekly.loc[~weekly["complete_target_week"], ["revenue_w", "cogs_w"]] = np.nan
    weekly["revenue_w_lv1_target"] = weekly["revenue_w"]
    weekly["cogs_w_lv1_target"] = weekly["cogs_w"]
    weekly["has_holiday"] = weekly["holiday_days"].fillna(0).gt(0).astype(float)

    weekly["time_index_week"] = np.arange(len(weekly), dtype=float)
    weekly["week_sin"] = np.sin(2 * np.pi * weekly["iso_week"] / 53.0)
    weekly["week_cos"] = np.cos(2 * np.pi * weekly["iso_week"] / 53.0)
    weekly["month_sin"] = np.sin(2 * np.pi * weekly["month"] / 12.0)
    weekly["month_cos"] = np.cos(2 * np.pi * weekly["month"] / 12.0)
    weekly["promo_active_days"] = weekly.get("active_promo_count", 0)
    weekly["promo_has_active"] = (weekly["promo_active_days"] > 0).astype(float)
    weekly["promo_flag"] = weekly["promo_has_active"]
    weekly["promo_discount_sum"] = weekly.get("avg_promo_discount_value", 0)
    use_projected_promo_for_lv1 = os.environ.get("FORECAST_LV1_USE_PROJECTED_PROMO", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if use_projected_promo_for_lv1:
        weekly["lv1_active_promo_count"] = weekly.get("active_promo_count", 0)
        weekly["lv1_avg_promo_discount_value"] = weekly.get("avg_promo_discount_value", 0)
    else:
        weekly["lv1_active_promo_count"] = weekly.get("active_promo_count_actual", weekly.get("active_promo_count", 0))
        weekly["lv1_avg_promo_discount_value"] = weekly.get(
            "avg_promo_discount_value_actual",
            weekly.get("avg_promo_discount_value", 0),
        )
    weekly["promo_start_days"] = weekly.get("promo_start_day", 0)
    weekly["promo_end_days"] = weekly.get("promo_end_day", 0)
    weekly["is_month_start_week"] = weekly.get("is_month_start", 0).gt(0).astype(float)
    weekly["is_month_end_week"] = weekly.get("is_month_end", 0).gt(0).astype(float)
    weekly["is_payday_week"] = weekly.get("is_payday_window", 0).gt(0).astype(float)
    weekly["is_payday_month_start_week"] = weekly.get("is_payday_month_start", 0).gt(0).astype(float)
    weekly["is_payday_month_end_week"] = weekly.get("is_payday_month_end", 0).gt(0).astype(float)
    weekly["is_payday_month_edge_week"] = weekly.get("is_payday_month_edge", 0).gt(0).astype(float)
    weekly["is_payday_promo_week"] = weekly.get("is_payday_promo", 0).gt(0).astype(float)
    weekly["is_tet_like_period"] = weekly.get("is_tet_window", 0).gt(0).astype(float)
    weekly["is_black_friday_like_period"] = weekly.get("is_black_friday_window", 0).gt(0).astype(float)
    for target in ["revenue_w", "cogs_w"]:
        lag = weekly[target].shift(1)
        ma4 = weekly[target].shift(1).rolling(4, min_periods=2).mean()
        weekly[f"{target}_shock_4w"] = (
            safe_div((lag - ma4).abs(), ma4)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(lower=0.0, upper=5.0)
        )

    return weekly.sort_values("week_start").reset_index(drop=True)
