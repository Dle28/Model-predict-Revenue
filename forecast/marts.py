from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from .common import Config, EPS, add_calendar_columns, read_csv, safe_div


def _slug(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_")
    return text or "unknown"


def _add_share_columns(df: pd.DataFrame, prefixes: Iterable[str], total_col: str) -> pd.DataFrame:
    out = df.copy()
    if total_col not in out.columns:
        return out
    denom = out[total_col].replace(0, np.nan)
    for prefix in prefixes:
        for col in [c for c in out.columns if c.startswith(prefix) and not c.startswith(f"{prefix}share_")]:
            share_col = f"{prefix}share_{col[len(prefix):]}"
            out[share_col] = safe_div(out[col], denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def aggregate_sales_daily(cfg: Config) -> pd.DataFrame:
    sales = read_csv(cfg.data_dir / "sales.csv", parse_dates=["Date"])
    out = sales.rename(columns={"Date": "date", "Revenue": "revenue", "COGS": "cogs"})
    out = out.groupby("date", as_index=False)[["revenue", "cogs"]].sum()
    return out


def aggregate_web_traffic_daily(cfg: Config) -> pd.DataFrame:
    wt = read_csv(cfg.data_dir / "web_traffic.csv", parse_dates=["date"])
    wt["bounce_x_sessions"] = wt["bounce_rate"] * wt["sessions"]
    wt["duration_x_sessions"] = wt["avg_session_duration_sec"] * wt["sessions"]
    grouped = wt.groupby("date", as_index=False).agg(
        sessions=("sessions", "sum"),
        unique_visitors=("unique_visitors", "sum"),
        page_views=("page_views", "sum"),
        bounce_x_sessions=("bounce_x_sessions", "sum"),
        duration_x_sessions=("duration_x_sessions", "sum"),
    )
    grouped["bounce_rate"] = safe_div(grouped["bounce_x_sessions"], grouped["sessions"])
    grouped["avg_session_duration_sec"] = safe_div(grouped["duration_x_sessions"], grouped["sessions"])
    grouped["pageviews_per_session"] = safe_div(grouped["page_views"], grouped["sessions"])
    grouped = grouped.drop(columns=["bounce_x_sessions", "duration_x_sessions"])

    source = (
        wt.pivot_table(index="date", columns="traffic_source", values="sessions", aggfunc="sum", fill_value=0)
        .add_prefix("sessions_source_")
        .reset_index()
    )
    out = grouped.merge(source, on="date", how="left")
    source_cols = [c for c in out.columns if c.startswith("sessions_source_")]
    if source_cols:
        total = out[source_cols].sum(axis=1).replace(0, np.nan)
        shares = out[source_cols].div(total, axis=0).fillna(0)
        out["source_entropy"] = -(shares * np.log(shares + EPS)).sum(axis=1)
        for col in source_cols:
            out[f"sessions_source_share_{col.removeprefix('sessions_source_')}"] = safe_div(out[col], out["sessions"]).fillna(0.0)
    return out


def aggregate_orders_daily(cfg: Config) -> pd.DataFrame:
    orders = read_csv(cfg.data_dir / "orders.csv", parse_dates=["order_date"])
    customers = read_csv(
        cfg.data_dir / "customers.csv",
        parse_dates=["signup_date"],
        usecols=["customer_id", "signup_date", "age_group", "acquisition_channel"],
    )
    orders = orders.merge(customers, on="customer_id", how="left")
    orders = orders.sort_values(["customer_id", "order_date", "order_id"]).copy()
    orders["order_seq"] = orders.groupby("customer_id").cumcount()
    orders["previous_order_date"] = orders.groupby("customer_id")["order_date"].shift()
    orders["inter_order_gap_days"] = (orders["order_date"] - orders["previous_order_date"]).dt.days
    orders["first_purchase_order"] = orders["order_seq"].eq(0)
    orders["repeat_purchase_order"] = orders["order_seq"].gt(0)
    orders["customer_tenure_days"] = (
        pd.to_datetime(orders["order_date"]).dt.normalize() - pd.to_datetime(orders["signup_date"]).dt.normalize()
    ).dt.days.clip(lower=0)
    orders["new_customer_order"] = pd.to_datetime(orders["order_date"]).dt.normalize().eq(
        pd.to_datetime(orders["signup_date"]).dt.normalize()
    )
    orders["new_order_customer_id"] = orders["customer_id"].where(orders["new_customer_order"])
    out = orders.groupby("order_date", as_index=False).agg(
        orders_count=("order_id", "nunique"),
        active_customers=("customer_id", "nunique"),
        new_order_customers=("new_order_customer_id", "nunique"),
        first_purchase_orders=("first_purchase_order", "sum"),
        repeat_purchase_orders=("repeat_purchase_order", "sum"),
        avg_inter_order_gap_days=("inter_order_gap_days", "mean"),
        median_inter_order_gap_days=("inter_order_gap_days", "median"),
        avg_customer_tenure_lag0=("customer_tenure_days", "mean"),
        cancelled_orders=("order_id", lambda s: 0),
    )
    status_counts = orders.pivot_table(
        index="order_date",
        columns="order_status",
        values="order_id",
        aggfunc="nunique",
        fill_value=0,
    ).reset_index()
    status_counts = status_counts.rename(columns={c: f"{c}_orders" for c in status_counts.columns if c != "order_date"})
    out = out.drop(columns=["cancelled_orders"]).merge(status_counts, on="order_date", how="left")

    for column, prefix in [
        ("order_source", "orders_source_"),
        ("device_type", "orders_device_"),
        ("payment_method", "orders_payment_"),
        ("age_group", "customer_age_group_"),
        ("acquisition_channel", "customer_acquisition_"),
    ]:
        counts = orders.pivot_table(
            index="order_date",
            columns=column,
            values="order_id",
            aggfunc="nunique",
            fill_value=0,
        ).reset_index()
        counts = counts.rename(columns={c: f"{prefix}{_slug(c)}" for c in counts.columns if c != "order_date"})
        out = out.merge(counts, on="order_date", how="left")

    out = out.rename(columns={"order_date": "date"})
    status_cols = [c for c in out.columns if c.endswith("_orders") and c != "orders_count"]
    for col in status_cols:
        out[col] = out[col].fillna(0)
    raw_count_prefixes = [
        "orders_source_",
        "orders_device_",
        "orders_payment_",
        "customer_age_group_",
        "customer_acquisition_",
    ]
    for col in [c for c in out.columns if any(c.startswith(prefix) for prefix in raw_count_prefixes)]:
        out[col] = out[col].fillna(0)
    out = _add_share_columns(out, raw_count_prefixes, "orders_count")
    out["cancel_rate"] = safe_div(out.get("cancelled_orders", 0), out["orders_count"])
    out["cod_order_share"] = safe_div(out.get("orders_payment_cod", 0), out["orders_count"]).fillna(0.0)
    out["fulfilled_orders"] = out.get("delivered_orders", 0) + out.get("shipped_orders", 0)
    out["fulfilled_rate"] = safe_div(out["fulfilled_orders"], out["orders_count"])
    out["new_customer_ratio"] = safe_div(out["new_order_customers"], out["active_customers"]).fillna(0.0)
    out["repeat_customers"] = (out["active_customers"] - out["new_order_customers"]).clip(lower=0)
    out["repeat_customer_ratio"] = safe_div(out["repeat_customers"], out["active_customers"]).fillna(0.0)
    out["repeat_order_share"] = safe_div(out["repeat_purchase_orders"], out["orders_count"]).fillna(0.0)
    out["orders_per_customer"] = safe_div(out["orders_count"], out["active_customers"]).fillna(0.0)
    out = out.rename(columns={"avg_customer_tenure_lag0": "avg_customer_tenure"})
    return out


def aggregate_basket_daily(cfg: Config) -> pd.DataFrame:
    orders = read_csv(cfg.data_dir / "orders.csv", parse_dates=["order_date"], usecols=["order_id", "order_date"])
    items = read_csv(cfg.data_dir / "order_items.csv")
    products = read_csv(cfg.data_dir / "products.csv", usecols=["product_id", "category", "price"])
    items["gross_before_discount"] = items["quantity"] * items["unit_price"]
    items["net_before_refund"] = (items["gross_before_discount"] - items["discount_amount"]).clip(lower=0)
    items["promo_line"] = items["promo_id"].notna() | items["promo_id_2"].notna()
    items["stacked_promo_line"] = items["promo_id"].notna() & items["promo_id_2"].notna()
    merged = items.merge(products, on="product_id", how="left").merge(orders, on="order_id", how="left")
    merged["promo_line_revenue"] = np.where(merged["promo_line"], merged["net_before_refund"], 0.0)
    merged["promo_line_discount_amount"] = np.where(merged["promo_line"], merged["discount_amount"], 0.0)
    out = merged.groupby("order_date", as_index=False).agg(
        units=("quantity", "sum"),
        gross_before_discount=("gross_before_discount", "sum"),
        discount_amount=("discount_amount", "sum"),
        net_item_revenue_before_refund=("net_before_refund", "sum"),
        promo_line_revenue=("promo_line_revenue", "sum"),
        promo_line_discount_amount=("promo_line_discount_amount", "sum"),
        promo_orders=("order_id", lambda s: s[merged.loc[s.index, "promo_line"]].nunique()),
        stacked_promo_orders=("order_id", lambda s: s[merged.loc[s.index, "stacked_promo_line"]].nunique()),
        avg_unit_price=("unit_price", "mean"),
        avg_catalog_price=("price", "mean"),
    )
    category = merged.pivot_table(
        index="order_date",
        columns="category",
        values="net_before_refund",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    category = category.rename(columns={c: f"category_revenue_{_slug(c)}" for c in category.columns if c != "order_date"})
    out = out.merge(category, on="order_date", how="left")
    out = out.rename(columns={"order_date": "date"})
    out["promo_order_share"] = safe_div(out["promo_orders"], merged.groupby("order_date")["order_id"].nunique().reindex(out["date"]).to_numpy()).fillna(0.0)
    out["stacked_promo_share"] = safe_div(out["stacked_promo_orders"], merged.groupby("order_date")["order_id"].nunique().reindex(out["date"]).to_numpy()).fillna(0.0)
    out["promo_discount_rate"] = safe_div(out["promo_line_discount_amount"], out["promo_line_revenue"] + out["promo_line_discount_amount"]).fillna(0.0)
    return out


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
        overstock_sku_days=("overstock_flag", "sum"),
        reorder_sku_days=("reorder_flag", "sum"),
        sell_through_rate=("sell_through_rate", "mean"),
    )
    return out.rename(columns={"snapshot_date": "date"})


def aggregate_shipments_daily(cfg: Config) -> pd.DataFrame:
    shp = read_csv(cfg.data_dir / "shipments.csv", parse_dates=["ship_date", "delivery_date"])
    shipped = shp.groupby("ship_date", as_index=False).agg(
        shipment_orders=("order_id", "nunique"),
        shipping_fee=("shipping_fee", "sum"),
    ).rename(columns={"ship_date": "date"})
    delivered = shp.groupby("delivery_date", as_index=False).agg(
        delivered_orders_by_delivery_date=("order_id", "nunique"),
    ).rename(columns={"delivery_date": "date"})
    return shipped.merge(delivered, on="date", how="outer")


def aggregate_returns_daily(cfg: Config) -> pd.DataFrame:
    ret = read_csv(cfg.data_dir / "returns.csv", parse_dates=["return_date"])
    out = ret.groupby("return_date", as_index=False).agg(
        return_records=("return_id", "nunique"),
        return_events=("order_id", "nunique"),
        returned_units=("return_quantity", "sum"),
        refund_amount=("refund_amount", "sum"),
        defective_returns=("return_reason", lambda s: (s == "defective").sum()),
    )
    return out.rename(columns={"return_date": "date"})


def aggregate_reviews_daily(cfg: Config) -> pd.DataFrame:
    rev = read_csv(cfg.data_dir / "reviews.csv", parse_dates=["review_date"])
    out = rev.groupby("review_date", as_index=False).agg(
        review_count=("review_id", "nunique"),
        avg_rating=("rating", "mean"),
        low_rating_count=("rating", lambda s: (s <= 2).sum()),
    )
    out = out.rename(columns={"review_date": "date"})
    out["low_rating_share"] = safe_div(out["low_rating_count"], out["review_count"])
    return out


def aggregate_customers_daily(cfg: Config) -> pd.DataFrame:
    customers = read_csv(cfg.data_dir / "customers.csv", parse_dates=["signup_date"])
    out = customers.groupby("signup_date", as_index=False).agg(
        new_customers=("customer_id", "nunique"),
    )
    return out.rename(columns={"signup_date": "date"})


def aggregate_promotions_daily(cfg: Config, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    promos = read_csv(cfg.data_dir / "promotions.csv", parse_dates=["start_date", "end_date"])
    dates = pd.DataFrame({"date": pd.date_range(start_dt, end_dt, freq="D")})
    promo_defaults = {
        "active_promo_count": 0,
        "avg_promo_discount_value": 0.0,
        "stackable_promo_count": 0,
        "promo_start_day": 0,
        "promo_end_day": 0,
        "promo_intensity_day": 0.0,
        "days_to_promo": 999.0,
        "days_since_promo_start": 999.0,
        "days_until_promo_end": 999.0,
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

    rows = []
    for row in promos.itertuples(index=False):
        promo_dates = pd.date_range(row.start_date, row.end_date, freq="D")
        promo_dates = promo_dates[(promo_dates >= start_dt) & (promo_dates <= end_dt)]
        if len(promo_dates) == 0:
            continue
        rows.append(
            pd.DataFrame(
                {
                    "date": promo_dates,
                    "promo_id": row.promo_id,
                    "discount_value": row.discount_value,
                    "stackable_flag": row.stackable_flag,
                    "promo_start_day": (promo_dates.normalize() == pd.Timestamp(row.start_date).normalize()).astype(int),
                    "promo_end_day": (promo_dates.normalize() == pd.Timestamp(row.end_date).normalize()).astype(int),
                }
            )
        )
    if not rows:
        for col, value in promo_defaults.items():
            dates[col] = value
        return dates

    expanded = pd.concat(rows, ignore_index=True)
    out = expanded.groupby("date", as_index=False).agg(
        active_promo_count=("promo_id", "nunique"),
        avg_promo_discount_value=("discount_value", "mean"),
        stackable_promo_count=("stackable_flag", "sum"),
        promo_start_day=("promo_start_day", "max"),
        promo_end_day=("promo_end_day", "max"),
    )
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
    for col, value in promo_defaults.items():
        if col in merged.columns:
            merged[col] = merged[col].fillna(value)
        else:
            merged[col] = value
    active = merged["active_promo_count"].fillna(0).gt(0)
    merged.loc[active, "days_to_promo"] = 0.0
    merged["promo_intensity_day"] = merged["avg_promo_discount_value"].where(active, 0.0)
    return merged


def merge_daily(spine: pd.DataFrame, marts: Iterable[pd.DataFrame]) -> pd.DataFrame:
    out = spine.copy()
    for mart in marts:
        if mart["date"].duplicated().any():
            dupes = mart.loc[mart["date"].duplicated(), "date"].head().tolist()
            raise ValueError(f"Daily mart has duplicate dates before join: {dupes}")
        out = out.merge(mart, on="date", how="left")
    return out


def build_daily_mart(cfg: Config, sample: pd.DataFrame) -> pd.DataFrame:
    sales_daily = aggregate_sales_daily(cfg)
    start_dt = sales_daily["date"].min()
    end_dt = sample["Date"].max()
    spine = pd.DataFrame({"date": pd.date_range(start_dt, end_dt, freq="D")})

    marts = [
        sales_daily,
        aggregate_web_traffic_daily(cfg),
        aggregate_orders_daily(cfg),
        aggregate_basket_daily(cfg),
        aggregate_inventory_daily(cfg),
        aggregate_shipments_daily(cfg),
        aggregate_returns_daily(cfg),
        aggregate_reviews_daily(cfg),
        aggregate_customers_daily(cfg),
        aggregate_promotions_daily(cfg, start_dt, end_dt),
    ]
    daily = merge_daily(spine, marts).sort_values("date").reset_index(drop=True)
    daily = add_calendar_columns(daily, "date")
    daily["time_index_day"] = np.arange(len(daily), dtype=float)

    zero_fill_prefixes = (
        "sessions_source_",
        "orders_source_",
        "orders_device_",
        "orders_payment_",
        "customer_age_group_",
        "customer_acquisition_",
        "category_revenue_",
    )
    zero_fill_exact = {
        "sessions",
        "unique_visitors",
        "page_views",
        "orders_count",
        "active_customers",
        "new_order_customers",
        "first_purchase_orders",
        "repeat_purchase_orders",
        "repeat_customers",
        "created_orders",
        "paid_orders",
        "shipped_orders",
        "delivered_orders",
        "returned_orders",
        "cancelled_orders",
        "fulfilled_orders",
        "units",
        "gross_before_discount",
        "discount_amount",
        "net_item_revenue_before_refund",
        "promo_line_revenue",
        "promo_line_discount_amount",
        "promo_orders",
        "stacked_promo_orders",
        "stock_on_hand",
        "units_received",
        "inventory_units_sold",
        "stockout_days",
        "stockout_sku_days",
        "active_sku_days",
        "overstock_sku_days",
        "reorder_sku_days",
        "shipment_orders",
        "shipping_fee",
        "delivered_orders_by_delivery_date",
        "return_records",
        "return_events",
        "returned_units",
        "refund_amount",
        "defective_returns",
        "review_count",
        "low_rating_count",
        "new_customers",
        "active_promo_count",
        "stackable_promo_count",
        "promo_start_day",
        "promo_end_day",
    }
    for col in daily.columns:
        if col in zero_fill_exact or any(col.startswith(prefix) for prefix in zero_fill_prefixes):
            daily[col] = daily[col].fillna(0)

    rate_cols = [
        "bounce_rate",
        "avg_session_duration_sec",
        "pageviews_per_session",
        "source_entropy",
        "cancel_rate",
        "cod_order_share",
        "fulfilled_rate",
        "days_of_supply",
        "fill_rate",
        "sell_through_rate",
        "avg_rating",
        "low_rating_share",
        "avg_promo_discount_value",
        "promo_intensity_day",
        "days_to_promo",
        "days_since_promo_start",
        "days_until_promo_end",
        "avg_customer_tenure",
        "avg_inter_order_gap_days",
        "median_inter_order_gap_days",
        "avg_unit_price",
        "avg_catalog_price",
        "promo_discount_rate",
    ]
    for col in rate_cols:
        if col in daily.columns:
            daily[col] = daily[col].fillna(0)

    daily = daily.copy()
    daily["items_per_order"] = safe_div(daily["units"], daily["orders_count"])
    daily["orders_per_customer"] = safe_div(daily["orders_count"], daily["active_customers"]).fillna(0.0)
    daily["discount_rate"] = safe_div(daily["discount_amount"], daily["gross_before_discount"])
    daily["discount_amount_per_order"] = safe_div(daily["discount_amount"], daily["orders_count"])
    daily["promo_intensity"] = safe_div(daily["promo_orders"], daily["orders_count"])
    daily["promo_order_share"] = safe_div(daily["promo_orders"], daily["orders_count"]).fillna(0.0)
    daily["stacked_promo_share"] = safe_div(daily["stacked_promo_orders"], daily["orders_count"]).fillna(0.0)
    daily["promo_discount_rate"] = safe_div(daily["promo_line_discount_amount"], daily["promo_line_revenue"] + daily["promo_line_discount_amount"]).fillna(0.0)
    daily["return_rate"] = safe_div(daily["returned_orders"], daily["delivered_orders"].replace(0, np.nan)).fillna(0)
    daily["stockout_rate"] = safe_div(daily["stockout_sku_days"], daily["active_sku_days"])
    daily["overstock_rate"] = safe_div(daily["overstock_sku_days"], daily["active_sku_days"]).fillna(0.0)
    daily["reorder_rate"] = safe_div(daily["reorder_sku_days"], daily["active_sku_days"]).fillna(0.0)
    daily["stock_pressure_index"] = daily["stockout_rate"].fillna(0.0) * daily["sessions"].fillna(0.0)
    daily["available_supply_index"] = daily["fill_rate"].fillna(0.0) * daily["days_of_supply"].fillna(0.0)
    daily["defective_ratio"] = safe_div(daily["defective_returns"], daily["return_events"]).fillna(0.0)
    daily["new_customer_ratio"] = safe_div(daily["new_order_customers"], daily["active_customers"]).fillna(0.0)
    daily["repeat_customer_ratio"] = safe_div(daily["repeat_customers"], daily["active_customers"]).fillna(0.0)
    daily["repeat_order_share"] = safe_div(daily["repeat_purchase_orders"], daily["orders_count"]).fillna(0.0)
    daily["cod_order_share"] = safe_div(daily.get("orders_payment_cod", 0), daily["orders_count"]).fillna(0.0)
    daily["traffic_ma_7d"] = daily["sessions"].shift(1).rolling(7, min_periods=3).mean()
    daily["traffic_growth_7d"] = (
        safe_div(daily["sessions"], daily["traffic_ma_7d"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(lower=0.0, upper=5.0)
    )
    daily["promo_flag"] = daily.get("active_promo_count", 0).fillna(0).gt(0).astype(float)
    daily["new_customer_momentum_14d"] = (
        daily["new_customers"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .shift(1)
        .rolling(14, min_periods=1)
        .sum()
        .fillna(0.0)
    )
    for prefix, total_col in [
        ("sessions_source_", "sessions"),
        ("orders_source_", "orders_count"),
        ("orders_device_", "orders_count"),
        ("orders_payment_", "orders_count"),
        ("customer_age_group_", "orders_count"),
        ("customer_acquisition_", "orders_count"),
    ]:
        raw_cols = [c for c in daily.columns if c.startswith(prefix) and not c.startswith(f"{prefix}share_")]
        for col in raw_cols:
            share_col = f"{prefix}share_{col[len(prefix):]}"
            if share_col not in daily.columns:
                daily[share_col] = safe_div(daily[col], daily[total_col]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

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
        "bounce_rate",
        "avg_session_duration_sec",
        "pageviews_per_session",
        "source_entropy",
        "cancel_rate",
        "cod_order_share",
        "fulfilled_rate",
        "days_of_supply",
        "fill_rate",
        "sell_through_rate",
        "avg_rating",
        "low_rating_share",
        "avg_promo_discount_value",
        "items_per_order",
        "discount_rate",
        "promo_intensity",
        "promo_order_share",
        "stacked_promo_share",
        "promo_discount_rate",
        "return_rate",
        "stockout_rate",
        "overstock_rate",
        "reorder_rate",
        "orders_per_customer",
        "avg_customer_tenure",
        "avg_inter_order_gap_days",
        "median_inter_order_gap_days",
        "avg_unit_price",
        "avg_catalog_price",
        "discount_amount_per_order",
        "defective_ratio",
        "new_customer_ratio",
        "repeat_customer_ratio",
        "repeat_order_share",
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
        return (
            col in mean_cols
            or "_share_" in col
            or col.endswith("_share")
            or col.endswith("_ratio")
            or col.startswith("sessions_source_share_")
            or col.startswith("orders_source_share_")
            or col.startswith("orders_device_share_")
            or col.startswith("orders_payment_share_")
            or col.startswith("customer_age_group_share_")
            or col.startswith("customer_acquisition_share_")
        )
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
    weekly["promo_start_days"] = weekly.get("promo_start_day", 0)
    weekly["promo_end_days"] = weekly.get("promo_end_day", 0)
    weekly["is_month_start_week"] = weekly.get("is_month_start", 0).gt(0).astype(float)
    weekly["is_month_end_week"] = weekly.get("is_month_end", 0).gt(0).astype(float)
    weekly["is_payday_week"] = weekly.get("is_payday_window", 0).gt(0).astype(float)
    weekly["is_tet_like_period"] = weekly.get("is_tet_window", 0).gt(0).astype(float)
    weekly["is_black_friday_like_period"] = weekly.get("is_black_friday_window", 0).gt(0).astype(float)
    weekly["items_per_order"] = safe_div(weekly.get("units", 0), weekly.get("orders_count", 0))
    weekly["discount_rate"] = safe_div(weekly.get("discount_amount", 0), weekly.get("gross_before_discount", 0))
    weekly["promo_intensity"] = safe_div(weekly.get("promo_orders", 0), weekly.get("orders_count", 0))
    weekly["promo_order_share"] = safe_div(weekly.get("promo_orders", 0), weekly.get("orders_count", 0)).fillna(0)
    weekly["stacked_promo_share"] = safe_div(weekly.get("stacked_promo_orders", 0), weekly.get("orders_count", 0)).fillna(0)
    weekly["promo_discount_rate"] = safe_div(
        weekly.get("promo_line_discount_amount", 0),
        weekly.get("promo_line_revenue", 0) + weekly.get("promo_line_discount_amount", 0),
    ).fillna(0)
    weekly["return_rate"] = safe_div(weekly.get("returned_orders", 0), weekly.get("delivered_orders", 0)).replace([np.inf, -np.inf], np.nan).fillna(0)
    weekly["stockout_rate"] = safe_div(weekly.get("stockout_sku_days", 0), weekly.get("active_sku_days", 0))
    weekly["overstock_rate"] = safe_div(weekly.get("overstock_sku_days", 0), weekly.get("active_sku_days", 0)).fillna(0)
    weekly["reorder_rate"] = safe_div(weekly.get("reorder_sku_days", 0), weekly.get("active_sku_days", 0)).fillna(0)
    weekly["orders_per_session"] = safe_div(weekly.get("orders_count", 0), weekly.get("sessions", 0))
    weekly["orders_per_customer"] = safe_div(weekly.get("orders_count", 0), weekly.get("active_customers", 0)).fillna(0)
    weekly["stock_pressure_index"] = weekly["stockout_rate"].fillna(0) * weekly.get("sessions", 0).fillna(0)
    weekly["available_supply_index"] = weekly["fill_rate"].fillna(0) * weekly["days_of_supply"].fillna(0)
    weekly["traffic_ma_4w"] = weekly.get("sessions", 0).shift(1).rolling(4, min_periods=2).mean()
    weekly["traffic_growth"] = (
        safe_div(weekly.get("sessions", 0), weekly["traffic_ma_4w"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(lower=0.0, upper=5.0)
    )
    weekly["traffic_vs_2015_2018_median"] = 1.0
    baseline_mask = weekly["iso_year"].between(2015, 2018)
    traffic_baseline = weekly.loc[baseline_mask, "sessions"].replace(0, np.nan).median()
    if pd.notna(traffic_baseline) and traffic_baseline > EPS:
        weekly["traffic_vs_2015_2018_median"] = (
            safe_div(weekly.get("sessions", 0), traffic_baseline)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(1.0)
            .clip(lower=0.0, upper=5.0)
    )
    weekly["defective_ratio"] = safe_div(weekly.get("defective_returns", 0), weekly.get("return_events", 0)).fillna(0)
    weekly["new_customer_ratio"] = safe_div(weekly.get("new_order_customers", 0), weekly.get("active_customers", 0)).fillna(0)
    weekly["repeat_customers"] = (weekly.get("active_customers", 0) - weekly.get("new_order_customers", 0)).clip(lower=0)
    weekly["repeat_customer_ratio"] = safe_div(weekly["repeat_customers"], weekly.get("active_customers", 0)).fillna(0)
    weekly["repeat_order_share"] = safe_div(weekly.get("repeat_purchase_orders", 0), weekly.get("orders_count", 0)).fillna(0)
    weekly["cod_order_share"] = safe_div(weekly.get("orders_payment_cod", 0), weekly.get("orders_count", 0)).fillna(0)
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
