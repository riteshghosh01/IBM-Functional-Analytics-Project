from typing import Optional, Tuple

import pandas as pd
import plotly.express as px


def _looks_like_datetime(series: pd.Series) -> bool:
    sample = series.dropna()
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    valid = parsed.notna().sum()
    return valid / max(len(sample), 1) >= 0.6


def _safe_to_numeric(series: pd.Series) -> pd.Series:
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return pd.to_numeric(series, errors="coerce")


def _column_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return one Series even when a source file contains duplicate headers."""
    value = df.loc[:, column]
    return value.iloc[:, 0] if isinstance(value, pd.DataFrame) else value


def _prepare_aggregation_frame(df: pd.DataFrame, x_col: str, y_col: str, agg: str, color_col: Optional[str] = None):
    """Build a chart-ready aggregation table that works for date, category, and numeric columns."""
    if x_col not in df.columns:
        return pd.DataFrame()

    temp = pd.DataFrame({x_col: _column_series(df, x_col)})
    if y_col:
        temp[y_col] = _column_series(df, y_col).to_numpy()
    if y_col:
        temp[y_col] = _safe_to_numeric(temp[y_col])

    if temp.empty:
        return pd.DataFrame()

    if _looks_like_datetime(temp[x_col]):
        temp[x_col] = pd.to_datetime(temp[x_col], errors="coerce")
        temp = temp.dropna(subset=[x_col])
        if temp.empty:
            return pd.DataFrame()
        if y_col:
            temp = temp.groupby(pd.Grouper(key=x_col, freq="M" if len(temp) > 60 else "D"), dropna=False)[y_col].agg(agg).reset_index()
        else:
            temp = temp.groupby(pd.Grouper(key=x_col, freq="M" if len(temp) > 60 else "D"), dropna=False).size().reset_index(name="count")
        return temp

    if color_col and color_col != "None" and color_col in df.columns and y_col:
        color_data = df[[x_col, color_col, y_col]].copy()
        color_data[y_col] = _safe_to_numeric(color_data[y_col])
        grouped = color_data.groupby([x_col, color_col], dropna=False)[y_col].agg(agg).reset_index()
        if grouped.empty:
            return pd.DataFrame()
        if grouped[x_col].nunique() > 20:
            top = grouped.groupby(x_col, dropna=False)[y_col].sum().sort_values(ascending=False).head(12).index.tolist()
            grouped = grouped[grouped[x_col].isin(top)].copy()
        return grouped

    if y_col:
        grouped = temp.groupby(x_col, dropna=False)[y_col].agg(agg).reset_index()
        if grouped.empty:
            return pd.DataFrame()
        if grouped[x_col].nunique() > 15:
            grouped = grouped.sort_values(by=y_col, ascending=False).head(15)
        return grouped

    counts = temp[x_col].value_counts(dropna=False).reset_index()
    counts.columns = [x_col, "count"]
    if counts.shape[0] > 15:
        counts = counts.head(15)
    return counts


def create_chart(
    df: pd.DataFrame,
    x_col: Optional[str],
    y_col: Optional[str],
    agg: str = "sum",
    chart_type: str = "bar",
    color_col: Optional[str] = None,
):
    """Create a robust chart that adapts to numeric, categorical, and date fields."""
    if df.empty or not x_col or x_col not in df.columns:
        return None

    if y_col is not None and y_col not in df.columns:
        y_col = None

    if y_col is None or y_col == "":
        counts = df[x_col].value_counts(dropna=False).reset_index()
        counts.columns = [x_col, "count"]
        if counts.shape[0] > 15:
            counts = counts.head(15)
        if chart_type == "pie":
            return px.pie(counts, names=x_col, values="count", title=f"Distribution of {x_col}")
        if chart_type == "line":
            return px.line(counts, x=x_col, y="count", title=f"Trend of {x_col}")
        return px.bar(counts, x=x_col, y="count", title=f"Distribution of {x_col}")

    x_series = _column_series(df, x_col)
    y_series = _column_series(df, y_col) if y_col else None

    if pd.api.types.is_numeric_dtype(y_series) and pd.api.types.is_numeric_dtype(x_series):
        if chart_type in {"scatter", "line"}:
            return px.scatter(
                df[[x_col, y_col]].dropna(),
                x=x_col,
                y=y_col,
                color=color_col if color_col and color_col != "None" and color_col in df.columns else None,
                title=f"{y_col} vs {x_col}",
                opacity=0.7,
            )
        if chart_type == "histogram":
            return px.histogram(df[[y_col]].dropna(), x=y_col, nbins=25, title=f"Distribution of {y_col}")
        if chart_type == "box":
            return px.box(
                df[[x_col, y_col]].dropna(),
                x=x_col,
                y=y_col,
                color=color_col if color_col and color_col != "None" and color_col in df.columns else None,
                title=f"Distribution of {y_col} by {x_col}",
            )

    if pd.api.types.is_numeric_dtype(y_series):
        grouped = _prepare_aggregation_frame(df, x_col, y_col, agg, color_col)
        if grouped.empty:
            return None

        if chart_type == "pie":
            pie_data = grouped.head(10)
            return px.pie(pie_data, names=x_col, values=y_col, title=f"{y_col} by {x_col}")
        if chart_type == "line":
            return px.line(grouped, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
        if chart_type == "box":
            return px.box(df[[x_col, y_col]].dropna(), x=x_col, y=y_col, title=f"{y_col} by {x_col}")
        return px.bar(
            grouped,
            x=x_col,
            y=y_col,
            color=color_col if color_col and color_col != "None" and color_col in df.columns else None,
            title=f"{y_col} by {x_col}",
        )

    if pd.api.types.is_numeric_dtype(x_series):
        if chart_type == "histogram":
            return px.histogram(df[[x_col]].dropna(), x=x_col, nbins=25, title=f"Distribution of {x_col}")
        if chart_type == "box":
            return px.box(df[[x_col, y_col]].dropna(), x=x_col, y=y_col, title=f"{y_col} by {x_col}")

    grouped = _prepare_aggregation_frame(df, x_col, y_col, agg, color_col)
    if grouped.empty:
        return None

    if chart_type == "pie":
        return px.pie(grouped, names=x_col, values=y_col, title=f"{y_col} by {x_col}")
    if chart_type == "line":
        return px.line(grouped, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
    return px.bar(
        grouped,
        x=x_col,
        y=y_col,
        color=color_col if color_col and color_col != "None" and color_col in df.columns else None,
        title=f"{y_col} by {x_col}",
    )


def create_correlation_heatmap(df: pd.DataFrame, numeric_cols):
    """Create a correlation heatmap for numeric columns."""
    if not numeric_cols or len(numeric_cols) < 2:
        return None
    corr = df[numeric_cols].corr(numeric_only=True)
    return px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Correlation Heatmap")


def create_time_series(df: pd.DataFrame, date_col: str, value_col: str):
    if date_col not in df.columns or value_col not in df.columns:
        return None
    temp = df[[date_col, value_col]].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col, value_col])
    if temp.empty:
        return None
    grouped = temp.groupby(pd.Grouper(key=date_col, freq="M"), dropna=False)[value_col].sum().reset_index()
    return px.line(grouped, x=date_col, y=value_col, title=f"Time Series for {value_col}")
