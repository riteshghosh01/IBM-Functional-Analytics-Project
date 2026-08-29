from typing import Dict, List

import pandas as pd
import plotly.express as px


def numeric_summary(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns."""
    if not cols:
        return pd.DataFrame()
    return df[cols].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].reset_index().rename(columns={"index": "column"})


def categorical_summary(df: pd.DataFrame, cols: List[str], top_n: int = 10) -> Dict[str, pd.DataFrame]:
    """Return top categories and frequency distribution for categorical columns."""
    summaries = {}
    for col in cols:
        series = df[col].dropna()
        if series.empty:
            summaries[col] = pd.DataFrame(columns=[col, "count"])
            continue
        summary = series.value_counts().head(top_n).reset_index()
        summary.columns = [col, "count"]
        summaries[col] = summary
    return summaries


def date_summary(df: pd.DataFrame, cols: List[str]) -> Dict[str, pd.DataFrame]:
    """Return monthly and yearly aggregations for date columns."""
    summaries = {}
    for col in cols:
        if col not in df.columns:
            continue
        try:
            s = pd.to_datetime(df[col], errors="coerce").dropna()
            if s.empty:
                continue
            monthly = s.dt.to_period("M").value_counts().sort_index().reset_index()
            monthly.columns = ["period", "count"]
            yearly = s.dt.to_period("Y").value_counts().sort_index().reset_index()
            yearly.columns = ["period", "count"]
            summaries[col] = {"monthly": monthly, "yearly": yearly}
        except Exception:
            continue
    return summaries


def make_histogram(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return None
    return px.histogram(df, x=col, nbins=20, title=f"Distribution of {col}")


def make_boxplot(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return None
    return px.box(df, y=col, title=f"Box Plot for {col}")


def make_bar_chart(df: pd.DataFrame, col: str, top_n: int = 10):
    if col not in df.columns:
        return None
    counts = df[col].dropna().value_counts().head(top_n).reset_index()
    counts.columns = [col, "count"]
    return px.bar(counts, x=col, y="count", title=f"Top {top_n} Categories for {col}")


def make_time_series(df: pd.DataFrame, date_col: str, value_col: str):
    if date_col not in df.columns or value_col not in df.columns:
        return None
    series = df[[date_col, value_col]].dropna().copy()
    series[date_col] = pd.to_datetime(series[date_col], errors="coerce")
    series = series.dropna(subset=[date_col])
    if series.empty:
        return None
    grouped = series.groupby(date_col)[value_col].sum().reset_index()
    return px.line(grouped, x=date_col, y=value_col, title=f"{value_col} Over Time")
