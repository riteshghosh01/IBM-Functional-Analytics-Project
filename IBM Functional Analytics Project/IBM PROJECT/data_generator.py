"""
Decision360 - Synthetic Data Generator
Generates a realistic daily orders dataset for the capstone demo.
Includes a deliberate seasonal trend, a product decline, and an
injected anomaly so the analytics/anomaly modules have something
real to detect (important for a convincing live demo).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

REGIONS = ["North", "South", "East", "West"]
PRODUCTS = ["Product A", "Product B", "Product C", "Product D"]
SEGMENTS = ["Champions", "Loyal", "New", "At Risk", "Lost"]
CHANNELS = ["Online", "Retail Store", "Partner"]

N_CUSTOMERS = 600
N_DAYS = 180  # ~6 months of daily data


def generate_dataset(n_days: int = N_DAYS, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_date = datetime.today() - timedelta(days=n_days)

    customer_ids = [f"CUST{1000+i}" for i in range(N_CUSTOMERS)]
    customer_segment = {c: rng.choice(SEGMENTS, p=[0.15, 0.25, 0.2, 0.25, 0.15]) for c in customer_ids}

    rows = []
    order_id = 10000

    for day_idx in range(n_days):
        date = start_date + timedelta(days=day_idx)

        # base demand with weekly seasonality + slow upward trend
        weekday_factor = 1.3 if date.weekday() in (4, 5) else 1.0
        trend_factor = 1 + (day_idx / n_days) * 0.25
        base_orders = rng.poisson(45 * weekday_factor * trend_factor)

        # Inject a deliberate decline in Product B sales over the last 30 days
        # (this is what the "Why did revenue drop" driver analysis will find)
        product_b_decline = 1.0
        if day_idx > n_days - 30:
            days_into_decline = day_idx - (n_days - 30)
            product_b_decline = max(0.35, 1 - days_into_decline * 0.03)

        # Inject one sharp anomaly: North region revenue crash on a specific day
        anomaly_day = n_days - 15
        north_anomaly_factor = 0.4 if day_idx == anomaly_day else 1.0

        for _ in range(base_orders):
            region = rng.choice(REGIONS, p=[0.3, 0.25, 0.25, 0.2])
            product = rng.choice(PRODUCTS, p=[0.35, 0.3, 0.2, 0.15])
            customer = rng.choice(customer_ids)
            channel = rng.choice(CHANNELS, p=[0.55, 0.35, 0.1])

            # skip some orders to simulate the product B decline
            if product == "Product B" and rng.random() > product_b_decline:
                continue
            # skip some orders to simulate the north region anomaly
            if region == "North" and rng.random() > north_anomaly_factor:
                continue

            unit_price = {"Product A": 500, "Product B": 750, "Product C": 300, "Product D": 1200}[product]
            quantity = rng.integers(1, 5)
            discount_pct = rng.choice([0, 5, 10, 15, 20], p=[0.4, 0.25, 0.2, 0.1, 0.05])
            revenue = unit_price * quantity * (1 - discount_pct / 100)
            cost = unit_price * quantity * 0.6  # 60% COGS assumption
            profit = revenue - cost

            rows.append({
                "order_id": order_id,
                "date": date.date(),
                "region": region,
                "product": product,
                "category": "Core" if product in ("Product A", "Product B") else "Accessory",
                "customer_id": customer,
                "customer_segment": customer_segment[customer],
                "channel": channel,
                "quantity": int(quantity),
                "unit_price": unit_price,
                "discount_pct": discount_pct,
                "revenue": round(revenue, 2),
                "cost": round(cost, 2),
                "profit": round(profit, 2),
            })
            order_id += 1

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def generate_inventory() -> pd.DataFrame:
    """Simple current-inventory snapshot for the Inventory Decision module."""
    rng = np.random.default_rng(7)
    rows = []
    for product in PRODUCTS:
        current_stock = rng.integers(50, 800)
        daily_demand = rng.integers(8, 40)
        lead_time_days = rng.integers(3, 12)
        rows.append({
            "product": product,
            "current_stock": int(current_stock),
            "avg_daily_demand": int(daily_demand),
            "lead_time_days": int(lead_time_days),
            "days_of_stock_left": round(current_stock / max(daily_demand, 1), 1),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("orders.csv", index=False)
    inv = generate_inventory()
    inv.to_csv("inventory.csv", index=False)
    print(f"Generated {len(df)} orders across {df['date'].nunique()} days.")
    print(df.head())
