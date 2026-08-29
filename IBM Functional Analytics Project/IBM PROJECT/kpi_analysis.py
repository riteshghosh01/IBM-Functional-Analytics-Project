from typing import Dict, Optional

import pandas as pd


def detect_candidate_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Guess likely business columns from the dataset."""
    cols = {col.lower(): col for col in df.columns}

    revenue_col = None
    for key in ["revenue", "sales", "amount", "total", "income"]:
        if key in cols:
            revenue_col = cols[key]
            break
    if revenue_col is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        revenue_col = numeric_cols[0] if numeric_cols else None

    profit_col = None
    for key in ["profit", "margin", "gain", "net_profit"]:
        if key in cols:
            profit_col = cols[key]
            break

    quantity_col = None
    for key in ["quantity", "qty", "units", "count"]:
        if key in cols:
            quantity_col = cols[key]
            break

    date_col = None
    for key in ["date", "timestamp", "order_date", "created_at", "time"]:
        if key in cols:
            date_col = cols[key]
            break

    category_col = None
    for key in ["category", "product", "item", "segment", "product_category"]:
        if key in cols:
            category_col = cols[key]
            break

    region_col = None
    for key in ["region", "state", "country", "market", "area"]:
        if key in cols:
            region_col = cols[key]
            break

    return {
        "revenue_col": revenue_col,
        "profit_col": profit_col,
        "quantity_col": quantity_col,
        "date_col": date_col,
        "category_col": category_col,
        "region_col": region_col,
    }


def calculate_dynamic_kpis(
    df: pd.DataFrame,
    revenue_col: Optional[str] = None,
    profit_col: Optional[str] = None,
    quantity_col: Optional[str] = None,
    date_col: Optional[str] = None,
) -> Dict[str, object]:
    """Compute dataset-appropriate KPI metrics given user-selected columns."""
    metrics: Dict[str, object] = {}

    if revenue_col and revenue_col in df.columns:
        revenue_series = pd.to_numeric(df[revenue_col], errors="coerce").dropna()
        metrics["total_revenue"] = float(revenue_series.sum())
        metrics["average_revenue"] = float(revenue_series.mean()) if not revenue_series.empty else 0.0
        metrics["max_revenue"] = float(revenue_series.max()) if not revenue_series.empty else 0.0
        metrics["min_revenue"] = float(revenue_series.min()) if not revenue_series.empty else 0.0

    if profit_col and profit_col in df.columns:
        profit_series = pd.to_numeric(df[profit_col], errors="coerce").dropna()
        metrics["total_profit"] = float(profit_series.sum())
        metrics["average_profit"] = float(profit_series.mean()) if not profit_series.empty else 0.0
        metrics["max_profit"] = float(profit_series.max()) if not profit_series.empty else 0.0
        metrics["min_profit"] = float(profit_series.min()) if not profit_series.empty else 0.0

    if quantity_col and quantity_col in df.columns:
        qty_series = pd.to_numeric(df[quantity_col], errors="coerce").dropna()
        metrics["total_quantity"] = float(qty_series.sum())
        metrics["average_quantity"] = float(qty_series.mean()) if not qty_series.empty else 0.0

    if revenue_col and revenue_col in df.columns and quantity_col and quantity_col in df.columns:
        qty = pd.to_numeric(df[quantity_col], errors="coerce").fillna(0)
        revenue_val = pd.to_numeric(df[revenue_col], errors="coerce").fillna(0)
        if qty.sum() > 0:
            metrics["avg_transaction_value"] = float(revenue_val.sum() / qty.sum())
        else:
            metrics["avg_transaction_value"] = 0.0

    if date_col and date_col in df.columns and revenue_col and revenue_col in df.columns:
        try:
            daily = df[[date_col, revenue_col]].copy()
            daily[date_col] = pd.to_datetime(daily[date_col], errors="coerce")
            daily = daily.dropna(subset=[date_col, revenue_col])
            if not daily.empty and daily[date_col].nunique() > 1:
                daily = daily.groupby(date_col)[revenue_col].sum().sort_index()
                first = daily.iloc[0] if len(daily) > 0 else 0
                last = daily.iloc[-1] if len(daily) > 0 else 0
                if first != 0:
                    growth = ((last - first) / abs(first)) * 100
                else:
                    growth = 0.0
                metrics["growth_rate"] = float(growth)
        except Exception:
            metrics["growth_rate"] = None

    return metrics
