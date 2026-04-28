from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .common import Config, EPS, add_calendar_columns, read_csv, safe_div


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
    return out


def aggregate_orders_daily(cfg: Config) -> pd.DataFrame:
    orders = read_csv(cfg.data_dir / "orders.csv", parse_dates=["order_date"])
    out = orders.groupby("order_date", as_index=False).agg(
        orders_count=("order_id", "nunique"),
        active_customers=("customer_id", "nunique"),
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

    source_counts = orders.pivot_table(
        index="order_date",
        columns="order_source",
        values="order_id",
        aggfunc="nunique",
        fill_value=0,
    ).reset_index()
    source_counts = source_counts.rename(columns={c: f"orders_source_/{c}".replace("/", "") for c in source_counts.columns if c != "order_date"})
    out = out.merge(source_counts, on="order_date", how="left")

    out = out.rename(columns={"order_date": "date"})
    status_cols = [c for c in out.columns if c.endswith("_orders") and c != "orders_count"]
    for col in status_cols:
        out[col] = out[col].fillna(0)
    out["cancel_rate"] = safe_div(out.get("cancelled_orders", 0), out["orders_count"])
    out["fulfilled_orders"] = out.get("delivered_orders", 0) + out.get("shipped_orders", 0)
    out["fulfilled_rate"] = safe_div(out["fulfilled_orders"], out["orders_count"])
    return out


def aggregate_basket_daily(cfg: Config) -> pd.DataFrame:
    orders = read_csv(cfg.data_dir / "orders.csv", parse_dates=["order_date"], usecols=["order_id", "order_date"])
    items = read_csv(cfg.data_dir / "order_items.csv")
    items["gross_before_discount"] = items["quantity"] * items["unit_price"]
    items["net_before_refund"] = (items["gross_before_discount"] - items["discount_amount"]).clip(lower=0)
    items["promo_line"] = items["promo_id"].notna() | items["promo_id_2"].notna()
    merged = items.merge(orders, on="order_id", how="left")
    out = merged.groupby("order_date", as_index=False).agg(
        units=("quantity", "sum"),
        gross_before_discount=("gross_before_discount", "sum"),
        discount_amount=("discount_amount", "sum"),
        net_item_revenue_before_refund=("net_before_refund", "sum"),
        promo_orders=("order_id", lambda s: s[merged.loc[s.index, "promo_line"]].nunique()),
    )
    out = out.rename(columns={"order_date": "date"})
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
        return_events=("order_id", "nunique"),
        returned_units=("return_quantity", "sum"),
        refund_amount=("refund_amount", "sum"),
        defective_returns=("return_reason", lambda s: (s == "defective").sum()),
        wrong_size_returns=("return_reason", lambda s: (s == "wrong_size").sum()),
        late_delivery_returns=("return_reason", lambda s: (s == "late_delivery").sum()),
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
    if promos.empty:
        dates["active_promo_count"] = 0
        dates["avg_promo_discount_value"] = 0.0
        dates["stackable_promo_count"] = 0
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
                }
            )
        )
    if not rows:
        dates["active_promo_count"] = 0
        dates["avg_promo_discount_value"] = 0.0
        dates["stackable_promo_count"] = 0
        return dates

    expanded = pd.concat(rows, ignore_index=True)
    out = expanded.groupby("date", as_index=False).agg(
        active_promo_count=("promo_id", "nunique"),
        avg_promo_discount_value=("discount_value", "mean"),
        stackable_promo_count=("stackable_flag", "sum"),
    )
    return dates.merge(out, on="date", how="left")


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
    daily = merge_daily(spine, marts)
    daily = add_calendar_columns(daily, "date")
    daily["time_index_day"] = np.arange(len(daily), dtype=float)

    zero_fill_prefixes = (
        "sessions_source_",
        "orders_source_",
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
        "fulfilled_orders",
        "units",
        "gross_before_discount",
        "discount_amount",
        "net_item_revenue_before_refund",
        "promo_orders",
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
        "return_events",
        "returned_units",
        "refund_amount",
        "defective_returns",
        "wrong_size_returns",
        "late_delivery_returns",
        "review_count",
        "low_rating_count",
        "new_customers",
        "active_promo_count",
        "stackable_promo_count",
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
        "fulfilled_rate",
        "days_of_supply",
        "fill_rate",
        "sell_through_rate",
        "avg_rating",
        "low_rating_share",
        "avg_promo_discount_value",
    ]
    for col in rate_cols:
        if col in daily.columns:
            daily[col] = daily[col].fillna(0)

    daily["aov"] = safe_div(daily["revenue"], daily["orders_count"]).replace([np.inf, -np.inf], np.nan)
    daily["items_per_order"] = safe_div(daily["units"], daily["orders_count"])
    daily["discount_rate"] = safe_div(daily["discount_amount"], daily["gross_before_discount"])
    daily["promo_intensity"] = safe_div(daily["promo_orders"], daily["orders_count"])
    daily["return_rate"] = safe_div(daily["returned_orders"], daily["delivered_orders"].replace(0, np.nan)).fillna(0)
    daily["refund_rate"] = safe_div(daily["refund_amount"], daily["revenue"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    daily["stockout_rate"] = safe_div(daily["stockout_sku_days"], daily["active_sku_days"])

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
        "fulfilled_rate",
        "days_of_supply",
        "fill_rate",
        "sell_through_rate",
        "avg_rating",
        "low_rating_share",
        "avg_promo_discount_value",
        "aov",
        "items_per_order",
        "discount_rate",
        "promo_intensity",
        "return_rate",
        "refund_rate",
        "stockout_rate",
    }
    excluded_calendar = {"iso_year", "iso_week", "iso_weekday", "month", "quarter", "day_of_month", "day_of_year"}
    agg = {
        col: ("mean" if col in mean_cols else "sum")
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
    weekly["complete_spine_week"] = weekly["days_in_spine"].eq(7)
    weekly["complete_target_week"] = weekly["revenue_days"].eq(7) & weekly["cogs_days"].eq(7)

    weekly = weekly.rename(columns={"revenue": "revenue_w", "cogs": "cogs_w"})
    weekly.loc[~weekly["complete_target_week"], ["revenue_w", "cogs_w"]] = np.nan

    weekly["time_index_week"] = np.arange(len(weekly), dtype=float)
    weekly["week_sin"] = np.sin(2 * np.pi * weekly["iso_week"] / 53.0)
    weekly["week_cos"] = np.cos(2 * np.pi * weekly["iso_week"] / 53.0)
    weekly["month_sin"] = np.sin(2 * np.pi * weekly["month"] / 12.0)
    weekly["month_cos"] = np.cos(2 * np.pi * weekly["month"] / 12.0)
    weekly["promo_active_days"] = weekly.get("active_promo_count", 0)
    weekly["promo_has_active"] = (weekly["promo_active_days"] > 0).astype(float)
    weekly["promo_discount_sum"] = weekly.get("avg_promo_discount_value", 0)
    weekly["aov"] = safe_div(weekly["revenue_w"], weekly.get("orders_count", 0)).replace([np.inf, -np.inf], np.nan)
    weekly["items_per_order"] = safe_div(weekly.get("units", 0), weekly.get("orders_count", 0))
    weekly["discount_rate"] = safe_div(weekly.get("discount_amount", 0), weekly.get("gross_before_discount", 0))
    weekly["promo_intensity"] = safe_div(weekly.get("promo_orders", 0), weekly.get("orders_count", 0))
    weekly["return_rate"] = safe_div(weekly.get("returned_orders", 0), weekly.get("delivered_orders", 0)).replace([np.inf, -np.inf], np.nan).fillna(0)
    weekly["refund_rate"] = safe_div(weekly.get("refund_amount", 0), weekly["revenue_w"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    weekly["stockout_rate"] = safe_div(weekly.get("stockout_sku_days", 0), weekly.get("active_sku_days", 0))
    weekly["orders_per_session"] = safe_div(weekly.get("orders_count", 0), weekly.get("sessions", 0))
    weekly["revenue_per_session"] = safe_div(weekly["revenue_w"], weekly.get("sessions", 0)).replace([np.inf, -np.inf], np.nan)
    weekly["cogs_ratio_w"] = safe_div(weekly["cogs_w"], weekly["revenue_w"]).replace([np.inf, -np.inf], np.nan)
    weekly["gross_margin_rate"] = 1.0 - weekly["cogs_ratio_w"]

    return weekly.sort_values("week_start").reset_index(drop=True)
