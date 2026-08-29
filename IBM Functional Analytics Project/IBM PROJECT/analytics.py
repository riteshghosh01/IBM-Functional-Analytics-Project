"""
Decision360 - Analytics Engine
Pure-Python/pandas/sklearn logic. No Streamlit dependency here so it
can be unit-tested independently of the UI layer.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------
# 1. KPI CALCULATION
# ---------------------------------------------------------------------

def compute_kpis(df: pd.DataFrame, period_days: int = 30) -> dict:
    """Compute headline KPIs for the last `period_days` vs the period before it."""
    max_date = df["date"].max()
    current_start = max_date - pd.Timedelta(days=period_days - 1)
    prev_start = current_start - pd.Timedelta(days=period_days)
    prev_end = current_start - pd.Timedelta(days=1)

    current = df[(df["date"] >= current_start) & (df["date"] <= max_date)]
    previous = df[(df["date"] >= prev_start) & (df["date"] <= prev_end)]

    def pct_change(cur, prev):
        if prev == 0:
            return 0.0
        return round((cur - prev) / prev * 100, 1)

    kpis = {}
    for label, col, agg in [
        ("revenue", "revenue", "sum"),
        ("profit", "profit", "sum"),
        ("orders", "order_id", "nunique"),
        ("customers", "customer_id", "nunique"),
    ]:
        cur_val = current[col].agg(agg)
        prev_val = previous[col].agg(agg)
        kpis[label] = {
            "current": round(float(cur_val), 2),
            "previous": round(float(prev_val), 2),
            "change_pct": pct_change(cur_val, prev_val),
        }

    # Average order value
    cur_aov = current["revenue"].sum() / max(current["order_id"].nunique(), 1)
    prev_aov = previous["revenue"].sum() / max(previous["order_id"].nunique(), 1)
    kpis["avg_order_value"] = {
        "current": round(cur_aov, 2),
        "previous": round(prev_aov, 2),
        "change_pct": pct_change(cur_aov, prev_aov),
    }

    # Simple churn proxy: customers active in previous period but not current
    prev_customers = set(previous["customer_id"].unique())
    cur_customers = set(current["customer_id"].unique())
    churned = prev_customers - cur_customers
    churn_rate = round(len(churned) / max(len(prev_customers), 1) * 100, 1)
    kpis["churn_rate"] = {"current": churn_rate, "previous": None, "change_pct": None}

    return kpis


# ---------------------------------------------------------------------
# 2. DRIVER / "WHY" ANALYSIS
# ---------------------------------------------------------------------

def driver_analysis(df: pd.DataFrame, metric: str = "revenue",
                     dimension: str = "region", period_days: int = 30) -> pd.DataFrame:
    """
    Explains a change in `metric` by breaking it down across `dimension`
    (e.g. region, product, customer_segment) comparing current vs previous period.
    Returns a dataframe sorted by contribution to the change, descending impact.
    """
    max_date = df["date"].max()
    current_start = max_date - pd.Timedelta(days=period_days - 1)
    prev_start = current_start - pd.Timedelta(days=period_days)
    prev_end = current_start - pd.Timedelta(days=1)

    current = df[(df["date"] >= current_start) & (df["date"] <= max_date)]
    previous = df[(df["date"] >= prev_start) & (df["date"] <= prev_end)]

    cur_group = current.groupby(dimension)[metric].sum()
    prev_group = previous.groupby(dimension)[metric].sum()

    combined = pd.DataFrame({"current": cur_group, "previous": prev_group}).fillna(0)
    combined["change"] = combined["current"] - combined["previous"]
    combined["change_pct"] = np.where(
        combined["previous"] != 0,
        (combined["change"] / combined["previous"] * 100).round(1),
        0.0,
    )
    total_change = combined["change"].sum()
    combined["contribution_pct"] = np.where(
        total_change != 0,
        (combined["change"] / total_change * 100).round(1),
        0.0,
    )
    return combined.sort_values("change").reset_index()


def explain_kpi_change(df: pd.DataFrame, metric: str = "revenue", period_days: int = 30) -> dict:
    """Runs driver analysis across all key dimensions and returns the top drivers."""
    dims = ["region", "product", "customer_segment", "channel"]
    top_drivers = []
    for dim in dims:
        result = driver_analysis(df, metric, dim, period_days)
        # take the single biggest mover (positive or negative) per dimension
        result["abs_change"] = result["change"].abs()
        biggest = result.sort_values("abs_change", ascending=False).iloc[0]
        top_drivers.append({
            "dimension": dim,
            "segment": biggest[dim],
            "change": round(float(biggest["change"]), 2),
            "change_pct": float(biggest["change_pct"]),
        })
    top_drivers.sort(key=lambda x: abs(x["change"]), reverse=True)
    return {"metric": metric, "period_days": period_days, "top_drivers": top_drivers}


# ---------------------------------------------------------------------
# 3. FORECASTING (simple, explainable — linear trend + weekday seasonality)
# ---------------------------------------------------------------------

def forecast_metric(df: pd.DataFrame, metric: str = "revenue", horizon_days: int = 14) -> pd.DataFrame:
    """
    Lightweight forecast using linear regression on day-index plus a
    weekday dummy adjustment. Deliberately simple and explainable rather
    than a black box — appropriate for a capstone project.
    """
    daily = df.groupby("date")[metric].sum().reset_index().sort_values("date")
    daily["day_idx"] = np.arange(len(daily))
    daily["weekday"] = daily["date"].dt.weekday

    weekday_avg = daily.groupby("weekday")[metric].mean()
    overall_avg = daily[metric].mean()
    weekday_effect = (weekday_avg - overall_avg).to_dict()

    X = daily[["day_idx"]].values
    y = daily[metric].values
    model = LinearRegression().fit(X, y)

    last_idx = daily["day_idx"].max()
    last_date = daily["date"].max()

    future_rows = []
    for i in range(1, horizon_days + 1):
        future_date = last_date + pd.Timedelta(days=i)
        future_idx = last_idx + i
        base_pred = model.predict([[future_idx]])[0]
        adj = weekday_effect.get(future_date.weekday(), 0)
        pred = max(base_pred + adj, 0)
        future_rows.append({"date": future_date, metric: pred, "type": "forecast"})

    history = daily[["date", metric]].copy()
    history["type"] = "actual"
    forecast_df = pd.DataFrame(future_rows)
    return pd.concat([history, forecast_df], ignore_index=True)


# ---------------------------------------------------------------------
# 4. ANOMALY DETECTION (z-score on daily metric, per dimension)
# ---------------------------------------------------------------------

def detect_anomalies(df: pd.DataFrame, metric: str = "revenue",
                      dimension: str = "region", z_threshold: float = 2.0) -> pd.DataFrame:
    """
    Flags dimension-days where the metric deviates more than z_threshold
    standard deviations from that dimension's rolling mean.
    """
    daily = df.groupby(["date", dimension])[metric].sum().reset_index()
    results = []
    for seg, group in daily.groupby(dimension):
        group = group.sort_values("date")
        mean = group[metric].mean()
        std = group[metric].std() or 1e-9
        group["z_score"] = (group[metric] - mean) / std
        anomalies = group[group["z_score"].abs() >= z_threshold]
        for _, row in anomalies.iterrows():
            results.append({
                "date": row["date"],
                dimension: seg,
                "actual": round(float(row[metric]), 2),
                "expected": round(float(mean), 2),
                "deviation_pct": round(float((row[metric] - mean) / mean * 100), 1) if mean else 0,
                "z_score": round(float(row["z_score"]), 2),
            })
    return pd.DataFrame(results).sort_values("date", ascending=False) if results else pd.DataFrame()


# ---------------------------------------------------------------------
# 5. RULE-BASED RECOMMENDATION ENGINE
# ---------------------------------------------------------------------

def generate_recommendations(df: pd.DataFrame, inventory_df: pd.DataFrame, period_days: int = 30) -> list:
    """
    Combines driver analysis + inventory status into prioritized,
    human-readable recommendations. Deliberately rule-based and
    transparent (no black-box "AI decided this") — this is what
    the report should describe as the Decision Engine.
    """
    recs = []

    # --- Product-level sales decline -> promotion recommendation
    product_drivers = driver_analysis(df, "revenue", "product", period_days)
    for _, row in product_drivers.iterrows():
        if row["change_pct"] <= -10:
            expected_revenue_gain = abs(row["change"]) * 0.5
            recs.append({
                "priority": "High",
                "title": f"{row['product']} sales declined {abs(row['change_pct'])}%",
                "action": f"Increase promotion/discount for {row['product']} in its weakest region.",
                "expected_impact_revenue": round(expected_revenue_gain, 2),
                "confidence": 72,
            })

    # --- Region-level anomaly -> investigate
    region_drivers = driver_analysis(df, "revenue", "region", period_days)
    for _, row in region_drivers.iterrows():
        if row["change_pct"] <= -15:
            recs.append({
                "priority": "High",
                "title": f"{row['region']} region revenue dropped {abs(row['change_pct'])}%",
                "action": f"Investigate {row['region']} region operations (site traffic, stockouts, local competition).",
                "expected_impact_revenue": round(abs(row["change"]) * 0.4, 2),
                "confidence": 65,
            })

    # --- Inventory-based reorder recommendations
    for _, row in inventory_df.iterrows():
        if row["days_of_stock_left"] <= row["lead_time_days"] * 1.2:
            reorder_qty = int(row["avg_daily_demand"] * (row["lead_time_days"] + 14))
            est_loss = row["avg_daily_demand"] * 500 * 3  # rough 3-day stockout estimate
            recs.append({
                "priority": "High" if row["days_of_stock_left"] <= row["lead_time_days"] else "Medium",
                "title": f"{row['product']} will likely stock out in {row['days_of_stock_left']} days",
                "action": f"Reorder {reorder_qty} units now (lead time is {row['lead_time_days']} days).",
                "expected_impact_revenue": round(est_loss, 2),
                "confidence": 80,
            })

    priority_rank = {"High": 0, "Medium": 1, "Low": 2}
    recs.sort(key=lambda r: priority_rank.get(r["priority"], 3))
    return recs


# ---------------------------------------------------------------------
# 6. WHAT-IF SIMULATOR (price / discount elasticity)
# ---------------------------------------------------------------------

def simulate_price_change(df: pd.DataFrame, product: str, price_change_pct: float,
                           elasticity: float = -1.4) -> dict:
    """
    Simple constant-elasticity simulator:
    %change in demand = elasticity * %change in price
    Elasticity default (-1.4) assumes moderately elastic demand;
    expose it as a parameter so the report can justify the assumption.
    """
    product_df = df[df["product"] == product]
    current_revenue = product_df["revenue"].sum()
    current_qty = product_df["quantity"].sum()
    current_avg_price = product_df["revenue"].sum() / max(current_qty, 1)

    demand_change_pct = elasticity * price_change_pct
    new_qty = current_qty * (1 + demand_change_pct / 100)
    new_price = current_avg_price * (1 + price_change_pct / 100)
    new_revenue = new_qty * new_price

    current_cost_ratio = product_df["cost"].sum() / max(product_df["revenue"].sum(), 1)
    new_profit = new_revenue * (1 - current_cost_ratio)
    current_profit = product_df["profit"].sum()

    return {
        "product": product,
        "price_change_pct": price_change_pct,
        "current_revenue": round(current_revenue, 2),
        "projected_revenue": round(new_revenue, 2),
        "revenue_change_pct": round((new_revenue - current_revenue) / max(current_revenue, 1) * 100, 1),
        "current_profit": round(current_profit, 2),
        "projected_profit": round(new_profit, 2),
        "profit_change_pct": round((new_profit - current_profit) / max(current_profit, 1) * 100, 1),
        "projected_demand_change_pct": round(demand_change_pct, 1),
    }


def simulate_discount_sweep(df: pd.DataFrame, product: str,
                             discounts=(0, 5, 10, 15, 20, 25), elasticity: float = -1.4) -> pd.DataFrame:
    """Runs the price simulator across a range of discount levels to find the revenue-optimal one."""
    rows = []
    for d in discounts:
        result = simulate_price_change(df, product, price_change_pct=-d, elasticity=elasticity)
        rows.append({"discount_pct": d, "projected_revenue": result["projected_revenue"],
                      "projected_profit": result["projected_profit"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 7. CUSTOMER SEGMENTATION (RFM)
# ---------------------------------------------------------------------

def rfm_segments(df: pd.DataFrame) -> pd.DataFrame:
    max_date = df["date"].max()
    rfm = df.groupby("customer_id").agg(
        recency=("date", lambda x: (max_date - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("revenue", "sum"),
    ).reset_index()

    rfm["r_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    def label(row):
        if row["rfm_score"] >= 13:
            return "Champions"
        elif row["rfm_score"] >= 10:
            return "Loyal Customers"
        elif row["rfm_score"] >= 7:
            return "New / Potential"
        elif row["rfm_score"] >= 5:
            return "At Risk"
        else:
            return "Lost"

    rfm["segment"] = rfm.apply(label, axis=1)
    return rfm
