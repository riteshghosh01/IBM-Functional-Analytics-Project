"""
Decision360 - Dynamic Business Intelligence & Decision Support Platform
The original project used a fixed sales dataset. This version upgrades it to
accept uploaded CSV/Excel datasets and automatically adapt analytics,
KPIs, visuals, and insights to the uploaded data structure.
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression

from analytics import (
    compute_kpis,
    detect_anomalies,
    explain_kpi_change,
    forecast_metric,
    generate_recommendations,
    rfm_segments,
    simulate_discount_sweep,
    simulate_price_change,
)
from business_insights import generate_business_insights
from data_cleaning import clean_dataset, detect_quality_issues
from data_generator import generate_dataset
from data_loader import get_dataset_overview, infer_column_types, load_uploaded_dataset
from eda import (
    categorical_summary,
    date_summary,
    make_bar_chart,
    make_boxplot,
    make_histogram,
    make_time_series,
    numeric_summary,
)
from kpi_analysis import calculate_dynamic_kpis, detect_candidate_cols
from predictive_analysis import build_and_evaluate_model
from visualization import create_chart, create_correlation_heatmap

st.set_page_config(page_title="Decision360", page_icon="D", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
    :root {
        --ink: #17202a;
        --muted: #68737d;
        --canvas: #f4f6f8;
        --surface: #ffffff;
        --line: #dfe5e8;
        --accent: #087f8c;
        --accent-soft: #e4f3f3;
        --success: #23835b;
        --warning: #b7791f;
    }
    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
    .stApp {
        background: linear-gradient(135deg, #f6f8f7 0%, #eef3f1 100%);
        color: var(--ink);
    }
    .main .block-container {
        max-width: 1500px;
        padding: 1.25rem 2.5rem 3rem;
    }
    div[data-testid="stSidebar"] {
        background: #fdfefe;
        border-right: 1px solid var(--line);
    }
    div[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
    div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--muted);
    }
    div[data-testid="stRadio"] label {
        border-radius: 8px;
        padding: 0.45rem 0.6rem;
        color: var(--muted);
    }
    div[data-testid="stRadio"] label:hover {
        background: var(--accent-soft);
    }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    h1 { font-size: 2rem !important; font-weight: 800 !important; }
    h2, h3 { font-weight: 700 !important; }
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        box-shadow: 0 4px 16px rgba(24, 39, 49, 0.04);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-size: 1.25rem !important;
        line-height: 1.2;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
    }
    .metric-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 16px rgba(24, 39, 49, 0.04);
    }
    .brand-mark { color: var(--accent); font-size: 1.35rem; font-weight: 800; letter-spacing: -0.03em; }
    .brand-note { color: var(--muted); font-size: 0.78rem; margin-bottom: 1.5rem; }
    .section-header {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 1rem;
    }
    .pill {
        display: inline-block;
        background: var(--accent-soft);
        border: 1px solid #b9dfe0;
        color: #17636a;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.78rem;
        margin: 0 0.25rem 0.35rem 0;
    }
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid var(--line);
        padding: 0.25rem 0 1.25rem;
        margin-bottom: 1.5rem;
    }
    .breadcrumb { color: var(--muted); font-size: 0.8rem; margin-bottom: 0.25rem; }
    .dataset-status { color: var(--success); font-size: 0.78rem; font-weight: 700; white-space: nowrap; }
    .dataset-status::before { content: '●'; margin-right: 0.35rem; }
    .eyebrow { color: var(--accent); font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; }
    .header-meta { color: var(--muted); font-size: 0.85rem; text-align: right; }
    .health-label { color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 800; }
    .stButton button, .stDownloadButton button, .stFormSubmitButton button {
        border-radius: 7px;
        border: 1px solid var(--accent);
        color: var(--accent);
        background: var(--surface);
        font-weight: 700;
    }
    .stButton button:hover, .stDownloadButton button:hover, .stFormSubmitButton button:hover {
        color: #ffffff;
        background: var(--accent);
        border-color: var(--accent);
    }
    [data-baseweb="tab-list"] { gap: 1.25rem; border-bottom: 1px solid var(--line); }
    [data-baseweb="tab"] {
        color: #000000 !important;
        font-weight: 700;
        opacity: 1 !important;
    }
    button[role="tab"] {
        color: #000000 !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }
    [data-baseweb="tab"][aria-selected="true"],
    button[role="tab"][aria-selected="true"] {
        color: #000000 !important;
    }
    div[data-baseweb="tab-list"] button,
    div[data-baseweb="tab-list"] button[aria-selected="false"],
    div[data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    div[data-baseweb="tab-list"] button span,
    div[data-baseweb="tab-list"] button p,
    div[data-baseweb="tab-list"] button div {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"],
    [data-testid="stTabs"] button[data-baseweb="tab"],
    [data-testid="stTabs"] [role="tab"] {
        color: #000000 !important;
        background-color: transparent !important;
        opacity: 1 !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] *,
    [data-testid="stTabs"] button[data-baseweb="tab"] *,
    [data-testid="stTabs"] [role="tab"] * {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        opacity: 1 !important;
    }
    [data-baseweb="tab"] p,
    [data-baseweb="tab"] div {
        color: #000000 !important;
    }
    button[role="tab"] * { color: #000000 !important; }
    [data-baseweb="tab-highlight"] { background: var(--accent); }
    @media (max-width: 800px) {
        .main .block-container { padding: 1rem 1rem 2rem; }
        .app-header { align-items: flex-start; gap: 0.75rem; flex-direction: column; }
        .header-meta { text-align: left; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_value(value):
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (int, float)):
        if abs(value) >= 1_000_000:
            return f"{value:,.0f}"
        if abs(value) >= 1_000:
            return f"{value:,.0f}"
        return f"{value:,.2f}"
    return str(value)


def business_numeric_columns(frame: pd.DataFrame) -> list:
    """Return numeric fields ordered by business relevance, excluding IDs."""
    numeric = frame.select_dtypes(include="number").columns.tolist()
    preferred = [
        "revenue", "sales", "profit", "quantity", "cost", "margin",
        "unit_price", "discount_pct", "amount", "total",
    ]
    useful = [col for col in numeric if not any(token in col.lower() for token in ["id", "index"])]
    return sorted(useful, key=lambda col: (preferred.index(col.lower()) if col.lower() in preferred else len(preferred), col.lower()))


def apply_global_filters(frame: pd.DataFrame, page_name: str) -> pd.DataFrame:
    """Apply compact shared filters to analytical pages only."""
    if frame.empty or page_name == "📁 Upload & Data Prep":
        return frame

    st.sidebar.markdown("<div class='health-label'>Filters</div>", unsafe_allow_html=True)
    filtered = frame.copy()
    filter_columns = [
        ("Region", next((c for c in frame.columns if c.lower() in {"region", "state", "country", "market", "area"}), None)),
        ("Category", next((c for c in frame.columns if c.lower() in {"category", "product_category"}), None)),
        ("Channel", next((c for c in frame.columns if c.lower() == "channel"), None)),
        ("Customer segment", next((c for c in frame.columns if c.lower() == "customer_segment"), None)),
    ]
    for label, column in filter_columns:
        if column:
            values = sorted(frame[column].dropna().astype(str).unique().tolist())
            selected = st.sidebar.multiselect(label, values, key=f"global_filter_{column}")
            if selected:
                filtered = filtered[filtered[column].astype(str).isin(selected)]

    date_column = next((c for c in frame.columns if c.lower() in {"date", "order_date", "timestamp", "created_at"}), None)
    if date_column:
        dates = pd.to_datetime(frame[date_column], errors="coerce").dropna()
        if not dates.empty:
            first_date = dates.min().normalize()
            last_date = dates.max().normalize()
            date_preset = st.sidebar.selectbox(
                "Date range",
                ["All dates", "Last 7 days", "Last 30 days", "Last 90 days", "Custom range"],
                key="global_date_preset",
                help="Choose a quick period or set your own start and end dates.",
            )
            if date_preset == "All dates":
                date_start, date_end = first_date, last_date
            elif date_preset == "Custom range":
                custom_range = st.sidebar.date_input(
                    "Choose dates",
                    value=(first_date.date(), last_date.date()),
                    min_value=first_date.date(),
                    max_value=last_date.date(),
                    key="global_custom_date_range",
                )
                if isinstance(custom_range, tuple) and len(custom_range) == 2:
                    date_start = pd.Timestamp(custom_range[0]).normalize()
                    date_end = pd.Timestamp(custom_range[1]).normalize()
                else:
                    date_start = date_end = pd.Timestamp(custom_range).normalize()
            else:
                days = int(date_preset.split()[1])
                date_start = max(first_date, last_date - pd.Timedelta(days=days - 1))
                date_end = last_date

            parsed_dates = pd.to_datetime(filtered[date_column], errors="coerce").dt.normalize()
            filtered = filtered[parsed_dates.between(date_start, date_end, inclusive="both")]
            st.sidebar.caption(f"{date_start.strftime('%d %b %Y')} to {date_end.strftime('%d %b %Y')}")

    if len(filtered) != len(frame):
        st.sidebar.caption(f"Showing {len(filtered):,} of {len(frame):,} records")
    return filtered


def data_quality_score(frame: pd.DataFrame) -> int:
    """Calculate a simple transparent quality score out of 100."""
    if frame.empty or not frame.shape[1]:
        return 0
    cells = max(frame.shape[0] * frame.shape[1], 1)
    completeness = 1 - (frame.isna().sum().sum() / cells)
    duplicate_penalty = min(frame.duplicated().sum() / max(len(frame), 1), 1)
    return max(0, round((completeness * 90) + ((1 - duplicate_penalty) * 10)))


def render_overview_badges(df):
    missing_count = int(df.isna().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    numeric_count = len(df.select_dtypes(include="number").columns)
    date_count = len([c for c in df.columns if pd.api.types.is_datetime64_any_dtype(pd.to_datetime(df[c], errors="coerce"))])

    complete_pct = 100 - (missing_count / max(len(df) * max(len(df.columns), 1), 1) * 100)
    cols = st.columns(4)
    cols[0].metric("Total records", f"{len(df):,}")
    cols[1].metric("Fields", f"{len(df.columns):,}", f"{numeric_count} numeric")
    cols[2].metric("Data complete", f"{complete_pct:.1f}%", "Excellent" if complete_pct >= 98 else "Review")
    cols[3].metric("Duplicate rows", f"{duplicate_count:,}", "Clean" if duplicate_count == 0 else "Review")

    st.markdown(
        f"""
        <div class="section-header">
            <span class="pill">Numeric: {numeric_count}</span>
            <span class="pill">Date: {date_count}</span>
            <span class="pill">Categorical: {len(column_types['categorical'])}</span>
            <span class="pill">Dataset: {len(df.columns)} fields</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# SESSION STATE FOR DYNAMIC DATASET HANDLING
# ---------------------------------------------------------------------

def default_dataset() -> pd.DataFrame:
    """Load the bundled sample dataset if no file has been uploaded."""
    orders_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orders.csv")
    if os.path.exists(orders_path):
        try:
            return pd.read_csv(orders_path)
        except Exception:
            pass

    df = generate_dataset()
    df.to_csv(orders_path, index=False)
    return df


if "raw_df" not in st.session_state:
    st.session_state.raw_df = default_dataset()
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = st.session_state.raw_df.copy()
if "column_types" not in st.session_state:
    st.session_state.column_types = infer_column_types(st.session_state.cleaned_df)
if "selected_kpis" not in st.session_state:
    st.session_state.selected_kpis = detect_candidate_cols(st.session_state.cleaned_df)
if "uploaded_signature" not in st.session_state:
    st.session_state.uploaded_signature = None


def reset_to_default():
    st.session_state.raw_df = default_dataset()
    st.session_state.cleaned_df = st.session_state.raw_df.copy()
    st.session_state.column_types = infer_column_types(st.session_state.cleaned_df)
    st.session_state.selected_kpis = detect_candidate_cols(st.session_state.cleaned_df)


def get_current_df() -> pd.DataFrame:
    return st.session_state.cleaned_df if "cleaned_df" in st.session_state else st.session_state.raw_df


# ---------------------------------------------------------------------
# SIDEBAR / NAVIGATION
# ---------------------------------------------------------------------

st.sidebar.markdown("<div class='brand-mark'>Decision360</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='brand-note'>Business intelligence & decision support</div>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel dataset",
    type=["csv", "xlsx", "xls"],
    help="Upload your own business dataset to replace the sample data.",
)

upload_signature = (uploaded_file.name, uploaded_file.size) if uploaded_file is not None else None
if uploaded_file is not None and upload_signature != st.session_state.uploaded_signature:
    try:
        df = load_uploaded_dataset(uploaded_file)
        st.session_state.raw_df = df
        st.session_state.cleaned_df = df.copy()
        st.session_state.column_types = infer_column_types(df)
        st.session_state.selected_kpis = detect_candidate_cols(df)
        st.session_state.uploaded_signature = upload_signature
        st.sidebar.success(f"Loaded dataset: {uploaded_file.name}")
    except Exception as exc:
        st.sidebar.error(f"Upload failed: {exc}")

if st.sidebar.button("Reset to sample dataset"):
    reset_to_default()

st.sidebar.markdown("<div class='health-label'>Workspace</div>", unsafe_allow_html=True)
workspace_pages = ["📁 Upload & Data Prep", "🏠 Executive Dashboard"]
st.sidebar.markdown("<div class='health-label'>Analytics</div>", unsafe_allow_html=True)
analytics_pages = ["📊 Sales Analytics", "📈 Visualization Studio", "👥 Customer Intelligence", "📦 Inventory Intelligence", "🔮 Forecasting"]
st.sidebar.markdown("<div class='health-label'>Decision tools</div>", unsafe_allow_html=True)
decision_pages = ["🚨 Anomaly Detection", "🎯 Decision Center", "🧪 What-If Simulator"]
page = st.sidebar.radio("Navigate", workspace_pages + analytics_pages + decision_pages, index=1, label_visibility="collapsed")

st.sidebar.markdown("---")
period_days = st.sidebar.slider("Analysis period (days)", 7, 60, 30)


base_df = get_current_df()
df = apply_global_filters(base_df, page)
for date_column in df.columns:
    if date_column.lower() in {"date", "order_date", "timestamp", "created_at", "time"}:
        parsed_date = pd.to_datetime(df[date_column], errors="coerce")
        if parsed_date.notna().any():
            df = df.copy()
            df[date_column] = parsed_date
column_types = infer_column_types(df)
numeric_cols = column_types["numeric"]
categorical_cols = column_types["categorical"]
date_cols = column_types["date"]

page_name = page.split(" ", 1)[1] if " " in page else page
st.markdown(
    f"""
    <div class="app-header">
        <div>
            <div class="breadcrumb">Decision360 / {page_name}</div>
            <div class="eyebrow">Decision intelligence workspace</div>
        </div>
        <div class="header-meta">
            Dataset: <strong>{len(df):,} records</strong> &nbsp; · &nbsp;
            <span class="dataset-status">Ready</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_overview_badges(df)

if df.empty:
    st.info("No records match the current filters. Clear a filter or upload a dataset to continue.")
    st.stop()

# ---------------------------------------------------------------------
# 1. DATA PREPARATION / CLEANING
# ---------------------------------------------------------------------

if page == "📁 Upload & Data Prep":
    st.title("Upload & Data Preparation")
    st.caption("Clean, validate, and understand your dataset before analysis.")

    overview = get_dataset_overview(df)
    quality = detect_quality_issues(df)
    missing_df = quality["missing_by_column"]
    dtype_df = pd.DataFrame({"column": list(overview["dtypes"].keys()), "dtype": list(overview["dtypes"].values())})

    prep_tabs = st.tabs(["Overview", "Columns", "Quality", "Preview"])
    with prep_tabs[0]:
        st.subheader("Dataset overview")
        st.write(f"Rows: {overview['rows']:,}  ·  Fields: {overview['columns']:,}")
        st.caption("The schema is inferred automatically from the active dataset.")
    with prep_tabs[1]:
        st.subheader("Detected columns")
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)
    with prep_tabs[2]:
        st.subheader("Data quality")
        quality_cols = st.columns(3)
        quality_cols[0].metric("Missing values", f"{int(df.isna().sum().sum()):,}")
        quality_cols[1].metric("Duplicate rows", f"{quality['duplicate_rows']:,}")
        quality_cols[2].metric("Empty columns", f"{len(quality['empty_columns']):,}")
        if missing_df.empty:
            st.success("No missing values detected.")
        else:
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
    with prep_tabs[3]:
        st.subheader("Dataset preview")
        st.dataframe(df.head(25), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Cleaning options")
    with st.form("cleaning_form"):
        remove_dups = st.checkbox("Remove duplicate rows", value=True)
        fill_missing_value = st.selectbox(
            "Missing value handling",
            ["mode", "median", "mean", "zero", "empty_string", "none"],
            index=0,
        )
        columns_to_remove = st.multiselect("Remove selected columns", df.columns.tolist())
        dt_cols = st.multiselect("Convert columns to datetime", [c for c in df.columns if c not in numeric_cols])

        submitted = st.form_submit_button("Apply cleaning")

    if submitted:
        cleaned = clean_dataset(
            df,
            drop_duplicates=remove_dups,
            fill_missing=fill_missing_value,
            remove_columns=columns_to_remove,
            convert_datetime_columns=dt_cols,
        )
        st.session_state.cleaned_df = cleaned
        st.session_state.column_types = infer_column_types(cleaned)
        st.success("Dataset cleaned successfully.")

    st.markdown("---")
    st.subheader("Cleaned dataset preview")
    st.dataframe(st.session_state.cleaned_df.head(10), use_container_width=True)

    st.download_button(
        label="Download cleaned CSV",
        data=st.session_state.cleaned_df.to_csv(index=False),
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------
# 2. EXECUTIVE DASHBOARD
# ---------------------------------------------------------------------

elif page == "🏠 Executive Dashboard":
    st.title("Executive Dashboard")
    st.caption("Dynamic KPI summary, trend analysis, and performance breakdowns from the uploaded data")

    if df.empty:
        st.warning("No dataset is available for analysis.")
    else:
        metric_candidates = business_numeric_columns(df)
        if not metric_candidates:
            metric_candidates = numeric_cols if numeric_cols else df.select_dtypes(include="number").columns.tolist()
        if not metric_candidates:
            st.warning("The uploaded dataset does not contain numeric fields for KPI analysis.")
        else:
            primary_metric = st.selectbox("Primary metric", metric_candidates, index=0)
            compare_metric = st.selectbox("Compare with (optional)", ["None"] + metric_candidates, index=0)
            breakdown_col = st.selectbox("Breakdown by", ["None"] + list(categorical_cols + date_cols), index=0)

            metric_cards = []
            for metric in metric_candidates[:4]:
                values = pd.to_numeric(df[metric], errors="coerce").dropna()
                if values.empty:
                    continue
                metric_cards.append((metric, float(values.sum()), float(values.mean()), float(values.max())))

            if metric_cards:
                card_cols = st.columns(min(len(metric_cards), 4))
                for col, (name, total, avg, maximum) in zip(card_cols, metric_cards):
                    with col:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown(f"### {name}")
                        st.metric("Total", f"{total:,.2f}")
                        st.caption(f"Avg: {avg:,.2f}   |   Max: {maximum:,.2f}")
                        st.markdown("</div>", unsafe_allow_html=True)

            insight_lines = []
            primary_series = pd.to_numeric(df[primary_metric], errors="coerce").dropna()
            if not primary_series.empty:
                total_val = float(primary_series.sum())
                avg_val = float(primary_series.mean())
                insight_lines.append(f"The {primary_metric} metric totals {total_val:,.2f} across the dataset, with an average of {avg_val:,.2f} per record.")

            if breakdown_col != "None" and breakdown_col in df.columns:
                grouped = df.groupby(breakdown_col, dropna=False)[primary_metric].sum().sort_values(ascending=False)
                if not grouped.empty:
                    top_name = grouped.index[0]
                    top_value = grouped.iloc[0]
                    insight_lines.append(f"{top_name} is the strongest segment for {primary_metric}, contributing {top_value:,.2f}.")

            if date_cols and primary_metric in df.columns:
                date_col = date_cols[0]
                trend = df[[date_col, primary_metric]].copy()
                trend[date_col] = pd.to_datetime(trend[date_col], errors="coerce")
                trend = trend.dropna(subset=[date_col, primary_metric])
                if not trend.empty:
                    trend = trend.groupby(pd.Grouper(key=date_col, freq="M"), dropna=False)[primary_metric].sum().reset_index()
                    if len(trend) > 1:
                        recent = float(trend[primary_metric].iloc[-1])
                        previous = float(trend[primary_metric].iloc[-2]) if len(trend) > 1 else recent
                        change = ((recent - previous) / previous * 100) if previous else 0.0
                        insight_lines.append(f"The latest monthly value is {recent:,.2f}, which is {change:+.2f}% versus the previous period.")

            st.markdown("### Automated analysis summary")
            for line in insight_lines[:4]:
                st.info(line)

            chart_col_1, chart_col_2 = st.columns(2)

            with chart_col_1:
                if breakdown_col != "None" and breakdown_col in df.columns and primary_metric in df.columns:
                    chart_df = df.groupby(breakdown_col, dropna=False)[primary_metric].sum().reset_index().sort_values(primary_metric, ascending=False).head(12)
                    fig = px.bar(chart_df, x=breakdown_col, y=primary_metric, color=breakdown_col, title=f"{primary_metric} by {breakdown_col}")
                    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.histogram(df[[primary_metric]].dropna(), x=primary_metric, nbins=30, title=f"Distribution of {primary_metric}")
                    st.plotly_chart(fig, use_container_width=True)

            with chart_col_2:
                if date_cols and primary_metric in df.columns:
                    date_col = date_cols[0]
                    trend = df[[date_col, primary_metric]].copy()
                    trend[date_col] = pd.to_datetime(trend[date_col], errors="coerce")
                    trend = trend.dropna(subset=[date_col, primary_metric]).groupby(pd.Grouper(key=date_col, freq="M"), dropna=False)[primary_metric].sum().reset_index()
                    if not trend.empty:
                        fig = px.line(trend, x=date_col, y=primary_metric, title=f"{primary_metric} trend over time")
                        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Date trend is unavailable for the selected metric.")
                else:
                    fig = px.box(df[[primary_metric]].dropna(), y=primary_metric, title=f"{primary_metric} distribution")
                    st.plotly_chart(fig, use_container_width=True)

            if compare_metric != "None" and compare_metric in df.columns and compare_metric != primary_metric:
                st.markdown("---")
                st.subheader("Comparative performance")
                comparison = df[[primary_metric, compare_metric]].dropna().copy()
                if not comparison.empty:
                    fig = px.scatter(comparison, x=primary_metric, y=compare_metric, title=f"{primary_metric} vs {compare_metric}")
                    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("Power BI overview")
            overview_tabs = st.tabs(["Performance", "Mix", "Distribution", "Relationships"])
            with overview_tabs[0]:
                if date_cols:
                    trend_col = date_cols[0]
                    trend_data = df[[trend_col, primary_metric]].copy()
                    trend_data[trend_col] = pd.to_datetime(trend_data[trend_col], errors="coerce")
                    trend_data[primary_metric] = pd.to_numeric(trend_data[primary_metric], errors="coerce")
                    trend_data = trend_data.dropna().groupby(pd.Grouper(key=trend_col, freq="M"))[primary_metric].sum().reset_index()
                    st.plotly_chart(px.area(trend_data, x=trend_col, y=primary_metric, title="Monthly performance"), use_container_width=True)
                else:
                    st.plotly_chart(px.histogram(df, x=primary_metric, title="Performance distribution"), use_container_width=True)
            with overview_tabs[1]:
                mix_col = categorical_cols[0] if categorical_cols else None
                if mix_col:
                    mix_data = df.groupby(mix_col, dropna=False)[primary_metric].sum().reset_index().sort_values(primary_metric, ascending=False).head(12)
                    st.plotly_chart(px.pie(mix_data, names=mix_col, values=primary_metric, title=f"{primary_metric} mix"), use_container_width=True)
                else:
                    st.info("Add a categorical field to see the business mix.")
            with overview_tabs[2]:
                st.plotly_chart(px.box(df, y=primary_metric, title=f"{primary_metric} spread"), use_container_width=True)
            with overview_tabs[3]:
                if len(numeric_cols) >= 2:
                    relation_cols = numeric_cols[:2]
                    st.plotly_chart(px.scatter(df, x=relation_cols[0], y=relation_cols[1], title="Metric relationships", opacity=0.65), use_container_width=True)
                else:
                    st.info("Add two numeric fields to see relationships.")

            if len(numeric_cols) >= 2:
                st.markdown("---")
                st.subheader("Correlation overview")
                heatmap = create_correlation_heatmap(df, numeric_cols[:8])
                if heatmap is not None:
                    st.plotly_chart(heatmap, use_container_width=True)

# ---------------------------------------------------------------------
# 3. SALES ANALYTICS
# ---------------------------------------------------------------------

elif page == "📊 Sales Analytics":
    st.title("Sales Analytics")
    numeric = numeric_cols or df.select_dtypes(include="number").columns.tolist()
    candidates = [col for col in df.columns if col in numeric or col in categorical_cols or col in date_cols]

    if not candidates:
        st.warning("The uploaded dataset does not contain fields suitable for sales analysis.")
    else:
        business_numeric = business_numeric_columns(df)
        selected_metric = st.selectbox("Metric", business_numeric if business_numeric else numeric if numeric else [df.columns[0]])
        selected_group = st.selectbox("Breakdown by", candidates, index=min(0, len(candidates)-1))

        filtered = df.copy()
        if selected_metric not in filtered.columns:
            st.warning("Please select a valid numeric metric.")
        else:
            agg_values = pd.to_numeric(filtered[selected_metric], errors="coerce")
            filtered["__metric__"] = agg_values
            chart_col_a, chart_col_b = st.columns(2)

            if selected_group in filtered.columns:
                grouped = filtered.groupby(selected_group, dropna=False)["__metric__"].agg(["sum", "mean", "count"]).reset_index()
                grouped = grouped.sort_values("sum", ascending=False).head(15)
                with chart_col_a:
                    st.subheader(f"{selected_metric} by {selected_group}")
                    st.dataframe(grouped.head(20), use_container_width=True)
                    st.plotly_chart(px.bar(grouped, x=selected_group, y="sum", title=f"{selected_metric} by {selected_group}"), use_container_width=True)

            if date_cols:
                date_col = date_cols[0]
                if date_col in filtered.columns:
                    daily = filtered[[date_col, "__metric__"]].copy()
                    daily[date_col] = pd.to_datetime(daily[date_col], errors="coerce")
                    daily = daily.dropna().groupby(pd.Grouper(key=date_col, freq="M" if len(daily) > 30 else "D"), dropna=False)["__metric__"].sum().reset_index()
                    with chart_col_b:
                        st.plotly_chart(px.line(daily, x=date_col, y="__metric__", title=f"{selected_metric} over time"), use_container_width=True)

            st.markdown("---")
            hist_fig = px.histogram(filtered["__metric__"].dropna(), nbins=25, title=f"Distribution of {selected_metric}")
            st.plotly_chart(hist_fig, use_container_width=True)

# ---------------------------------------------------------------------
# 4. VISUALIZATION STUDIO
# ---------------------------------------------------------------------

elif page == "📈 Visualization Studio":
    st.title("Visualization Studio")
    st.caption("Interactive charts based on the uploaded dataset and your selected fields")

    if df.empty:
        st.warning("No dataset available.")
    else:
        x_col = st.selectbox("X-axis", [col for col in df.columns], index=0)
        y_col = st.selectbox("Y-axis", ["None"] + [col for col in df.columns], index=1 if len(df.columns) > 1 else 0)
        agg_method = st.selectbox("Aggregation", ["sum", "mean", "count", "max", "min"], index=0)
        chart_type = st.selectbox("Chart type", ["bar", "line", "pie", "scatter", "histogram", "box"], index=0)
        color_col = st.selectbox("Color by (optional)", ["None"] + list(df.columns), index=0)

        color_value = None if color_col == "None" else color_col
        y_value = None if y_col == "None" else y_col
        chart = create_chart(df, x_col, y_value, agg=agg_method, chart_type=chart_type, color_col=color_value)
        if chart is not None:
            st.plotly_chart(chart, use_container_width=True)
            if y_value:
                summary_df = create_chart(df, x_col, y_value, agg=agg_method, chart_type="bar", color_col=color_value)
                if summary_df is not None:
                    st.caption(f"Chart summary: {x_col} vs {y_value} using {agg_method} aggregation")
                    st.dataframe(summary_df.data if hasattr(summary_df, 'data') else summary_df, use_container_width=True)
        else:
            st.info("Select valid columns to generate the chart.")

        if len(numeric_cols) >= 2:
            st.markdown("---")
            st.subheader("Correlation heatmap")
            heatmap = create_correlation_heatmap(df, numeric_cols[:10])
            if heatmap is not None:
                st.plotly_chart(heatmap, use_container_width=True)

# ---------------------------------------------------------------------
# 5. CUSTOMER INTELLIGENCE
# ---------------------------------------------------------------------

elif page == "👥 Customer Intelligence":
    st.title("Customer Intelligence")
    st.caption("Understand customer value, loyalty, and retention opportunities.")

    customer_candidates = [c for c in df.columns if "customer" in c.lower() or "client" in c.lower() or "user" in c.lower()]
    value_candidates = [c for c in numeric_cols if c.lower() not in {"id", "index"}]
    date_candidates = date_cols

    customer_options = customer_candidates or categorical_cols or [df.columns[0]]
    customer_col = st.selectbox("Customer ID column", customer_options)
    value_col = st.selectbox("Monetary value column", value_candidates if value_candidates else [df.columns[0]])
    date_options = date_candidates or ["Row sequence"]
    date_col = st.selectbox("Date column", date_options)

    canonical_rfm = {"customer_id", "date", "order_id", "revenue"}.issubset(df.columns)
    if canonical_rfm:
        try:
            customer_summary = rfm_segments(df.copy())
            segment_counts = customer_summary["segment"].value_counts().rename_axis("segment").reset_index(name="customers")
            metric_cols = st.columns(3)
            metric_cols[0].metric("Customers", f"{len(customer_summary):,}")
            metric_cols[1].metric("Champions", f"{int((customer_summary['segment'] == 'Champions').sum()):,}")
            metric_cols[2].metric("At risk / lost", f"{int(customer_summary['segment'].isin(['At Risk', 'Lost']).sum()):,}")
            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.plotly_chart(px.bar(segment_counts, x="segment", y="customers", color="segment", title="Customer segments"), use_container_width=True)
            with chart_cols[1]:
                st.plotly_chart(px.scatter(customer_summary, x="frequency", y="monetary", color="segment", hover_name="customer_id", title="Value and loyalty map"), use_container_width=True)
            st.dataframe(customer_summary.sort_values("monetary", ascending=False).head(25), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.warning(f"Customer segmentation unavailable: {exc}")
    elif customer_col and value_col and date_col:
        try:
            rfm = df[[customer_col, value_col]].copy()
            rfm[customer_col] = rfm[customer_col].fillna("Unknown").astype(str)
            rfm[value_col] = pd.to_numeric(rfm[value_col], errors="coerce")
            if date_col == "Row sequence":
                rfm[date_col] = pd.date_range(end=pd.Timestamp.today().normalize(), periods=len(rfm), freq="D")
            else:
                rfm[date_col] = pd.to_datetime(df[date_col], errors="coerce").to_numpy()
            rfm = rfm.dropna()
            if rfm.empty:
                st.info("Not enough clean data to build the customer segmentation view.")
            else:
                max_date = rfm[date_col].max()
                customer_summary = rfm.groupby(customer_col).agg(
                    recency=(date_col, lambda x: (max_date - x.max()).days),
                    frequency=(customer_col, "count"),
                    monetary=(value_col, "sum"),
                ).reset_index()
                customer_summary["segment"] = "General"
                st.dataframe(customer_summary.sort_values("monetary", ascending=False).head(20), use_container_width=True)
        except Exception as exc:
            st.warning(f"Customer intelligence unavailable: {exc}")
    else:
        st.info("This dataset does not contain obvious customer, date, and value columns for segmentation.")

# ---------------------------------------------------------------------
# 6. INVENTORY INTELLIGENCE
# ---------------------------------------------------------------------

elif page == "📦 Inventory Intelligence":
    st.title("Inventory Intelligence")

    inventory_like = [
        col for col in df.columns
        if any(keyword in col.lower() for keyword in ["stock", "inventory", "qty", "quantity", "supply", "available"])
    ]
    if not inventory_like:
        st.info("No stock/inventory-style columns were detected. Upload a dataset with inventory, stock, or quantity data for this section.")
    else:
        stock_col = st.selectbox("Stock/Inventory column", inventory_like)
        demand_candidates = [
            col for col in numeric_cols
            if col != stock_col and col.lower() not in {"id", "index"}
        ]
        demand_col = st.selectbox("Demand column (optional)", ["None"] + demand_candidates)
        if demand_col and demand_col != "None":
            inventory_df = df[[stock_col, demand_col]].copy()
            inventory_df[stock_col] = pd.to_numeric(inventory_df[stock_col], errors="coerce")
            inventory_df[demand_col] = pd.to_numeric(inventory_df[demand_col], errors="coerce")
            inventory_df["days_of_stock_left"] = inventory_df[stock_col] / inventory_df[demand_col].replace(0, np.nan)
            inventory_df = inventory_df.dropna(subset=[stock_col, demand_col])
            st.dataframe(inventory_df.head(20), use_container_width=True)
        else:
            st.dataframe(df[[stock_col]].head(20), use_container_width=True)

# ---------------------------------------------------------------------
# 7. FORECASTING
# ---------------------------------------------------------------------

elif page == "🔮 Forecasting":
    st.title("Forecasting")

    if not numeric_cols:
        st.info("Forecasting needs at least one numeric field in the active dataset.")
    else:
        forecast_candidates = business_numeric_columns(df) or numeric_cols
        value_col = st.selectbox("Metric to forecast", forecast_candidates)
        date_col = st.selectbox("Date column", date_cols or ["Row sequence"])
        horizon = st.slider("Forecast horizon (days)", 7, 60, 14)

        try:
            if {"date", value_col}.issubset(df.columns) and value_col in {"revenue", "profit", "quantity"}:
                forecast_df = forecast_metric(df[["date", value_col]].copy(), metric=value_col, horizon_days=horizon)
                st.plotly_chart(px.line(forecast_df, x="date", y=value_col, color="type", title=f"{value_col.title()} outlook"), use_container_width=True)
                projected = forecast_df.loc[forecast_df["type"] == "forecast", value_col].sum()
                st.metric("Projected total", f"{projected:,.2f}")
                st.dataframe(forecast_df.tail(horizon), use_container_width=True, hide_index=True)
                st.stop()

            daily = df[[value_col]].copy()
            daily[value_col] = pd.to_numeric(daily[value_col], errors="coerce")
            if date_col == "Row sequence":
                daily[date_col] = pd.date_range(end=pd.Timestamp.today().normalize(), periods=len(daily), freq="D")
            else:
                daily[date_col] = pd.to_datetime(df[date_col], errors="coerce").to_numpy()
            daily = daily.dropna().groupby(date_col, as_index=False)[value_col].sum()
            daily = daily.sort_values(date_col).reset_index(drop=True)
            daily["day_idx"] = np.arange(len(daily))
            model = LinearRegression()
            model.fit(daily[["day_idx"]], daily[value_col])

            last_date = daily[date_col].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
            future_idx = np.arange(len(daily), len(daily) + horizon)
            preds = model.predict(future_idx.reshape(-1, 1))
            forecast_df = pd.DataFrame({date_col: future_dates, value_col: preds})

            actual_df = daily[[date_col, value_col]].copy()
            actual_df["type"] = "actual"
            forecast_df["type"] = "forecast"
            combined = pd.concat([actual_df, forecast_df], ignore_index=True)

            st.plotly_chart(px.line(combined, x=date_col, y=value_col, color="type"), use_container_width=True)
            st.metric("Projected total", f"{forecast_df[value_col].sum():,.2f}")
        except Exception as exc:
            st.info(f"Forecast is waiting for more usable date and numeric data: {exc}")

# ---------------------------------------------------------------------
# 8. ANOMALY DETECTION
# ---------------------------------------------------------------------

elif page == "🚨 Anomaly Detection":
    st.title("Anomaly Detection")

    metric_col = st.selectbox("Metric to analyze", numeric_cols if numeric_cols else [df.columns[0]])
    dimension_col = st.selectbox("Breakdown by", categorical_cols if categorical_cols else [df.columns[0]])
    z_threshold = st.slider("Sensitivity (z-score threshold)", 1.5, 3.5, 2.0, step=0.1)

    if metric_col and dimension_col:
        try:
            anomalies = detect_anomalies(df, metric_col, dimension_col, z_threshold=z_threshold)
            if anomalies.empty:
                st.success("No anomalies detected at the current sensitivity.")
            else:
                st.dataframe(anomalies, use_container_width=True)
                st.warning(f"{len(anomalies)} records flagged as unusual.")
        except Exception as exc:
            st.info(f"Anomaly detection is waiting for compatible data: {exc}")

# ---------------------------------------------------------------------
# 9. DECISION CENTER
# ---------------------------------------------------------------------

elif page == "🎯 Decision Center":
    st.title("Decision Center")
    st.caption("Recommendations generated from the active dataset and selected KPI fields")

    recommendation_df = df.copy()
    revenue_col = st.selectbox("Revenue / sales metric", numeric_cols if numeric_cols else [df.columns[0]])
    category_col = st.selectbox("Category column (optional)", ["None"] + list(df.columns), index=0)
    region_col = st.selectbox("Region column (optional)", ["None"] + list(df.columns), index=0)
    date_col = st.selectbox("Date column (optional)", ["None"] + list(date_cols) + list(df.columns), index=0)

    insights = generate_business_insights(
        recommendation_df,
        revenue_col if isinstance(revenue_col, str) else None,
        category_col if category_col != "None" else None,
        region_col if region_col != "None" else None,
        date_col if date_col != "None" else None,
    )

    if not insights:
        st.info("No automated business insights could be generated for the selected fields.")
    else:
        for item in insights[:6]:
            st.markdown(item["text"])
            st.write(item["recommendation"])

# ---------------------------------------------------------------------
# 10. WHAT-IF SIMULATOR
# ---------------------------------------------------------------------

elif page == "🧪 What-If Simulator":
    st.title("What-If Simulator")
    st.caption("Adjust a business assumption and compare the projected outcome with today.")

    numeric = numeric_cols or df.select_dtypes(include="number").columns.tolist()
    if {"product", "revenue", "quantity", "cost", "profit"}.issubset(df.columns):
        product = st.selectbox("Product", sorted(df["product"].dropna().astype(str).unique()))
        price_change = st.slider("Price change (%)", -30, 30, 0, step=1)
        elasticity = st.slider("Demand elasticity", -2.5, -0.2, -1.4, step=0.1)
        scenario = simulate_price_change(df, product, price_change, elasticity)
        comparison = pd.DataFrame({
            "metric": ["Revenue", "Profit"],
            "Current": [scenario["current_revenue"], scenario["current_profit"]],
            "Projected": [scenario["projected_revenue"], scenario["projected_profit"]],
        })
        metric_cols = st.columns(3)
        metric_cols[0].metric("Projected revenue", f"{scenario['projected_revenue']:,.2f}", f"{scenario['revenue_change_pct']:+.1f}%")
        metric_cols[1].metric("Projected profit", f"{scenario['projected_profit']:,.2f}", f"{scenario['profit_change_pct']:+.1f}%")
        metric_cols[2].metric("Demand change", f"{scenario['projected_demand_change_pct']:+.1f}%")
        st.plotly_chart(px.bar(comparison, x="metric", y=["Current", "Projected"], barmode="group", title=f"Scenario impact: {product}"), use_container_width=True)
        sweep = simulate_discount_sweep(df, product, elasticity=elasticity)
        st.plotly_chart(px.line(sweep, x="discount_pct", y=["projected_revenue", "projected_profit"], markers=True, title="Discount sensitivity"), use_container_width=True)
        st.stop()

    metric_col = st.selectbox("Metric to simulate", business_numeric_columns(df) or numeric if numeric else [df.columns[0]])
    base_change = st.slider("Scenario change (%)", -50, 50, 0, step=1)

    current_total = pd.to_numeric(df[metric_col], errors="coerce").fillna(0).sum()
    new_total = current_total * (1 + (base_change / 100))

    st.metric("Current total", f"{current_total:,.2f}")
    st.metric("Projected total", f"{new_total:,.2f}", f"{base_change:+.0f}%")

    scenario_data = pd.DataFrame({
        "scenario": ["Current", "Projected"],
        "value": [current_total, new_total],
    })
    st.plotly_chart(px.bar(scenario_data, x="scenario", y="value", title="Scenario impact"), use_container_width=True)


# ---------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.caption(f"Active dataset: {len(df)} rows x {len(df.columns)} columns")
