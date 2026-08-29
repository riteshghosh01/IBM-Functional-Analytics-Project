import io
import os
from typing import Dict, List, Tuple

import pandas as pd


def detect_file_type(file_obj) -> str:
    """Return normalized file extension for uploaded data files."""
    if file_obj is None:
        raise ValueError("No file was uploaded.")
    name = getattr(file_obj, "name", "")
    ext = os.path.splitext(name)[1].lower()
    if ext in {".csv"}:
        return "csv"
    if ext in {".xls", ".xlsx"}:
        return "excel"
    raise ValueError("Unsupported file type. Please upload CSV or Excel (.xls/.xlsx).")


def load_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    """Load a CSV or Excel file from the upload widget."""
    if uploaded_file is None:
        raise ValueError("Upload a CSV or Excel file.")

    file_type = detect_file_type(uploaded_file)
    uploaded_file.seek(0)

    try:
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as exc:
        raise ValueError(f"Could not read the uploaded file: {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded dataset is empty.")

    df = df.copy()
    cleaned_columns = []
    seen_columns = {}
    for column in df.columns:
        name = str(column).strip() or "unnamed_column"
        occurrence = seen_columns.get(name, 0)
        seen_columns[name] = occurrence + 1
        cleaned_columns.append(name if occurrence == 0 else f"{name}_{occurrence + 1}")
    df.columns = cleaned_columns
    return df


def infer_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Identify numeric, categorical, text, and date-related columns."""
    numeric_cols = []
    categorical_cols = []
    date_cols = []
    text_cols = []

    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series) or str(series.dtype).lower().startswith("datetime"):
            date_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
            continue

        if series.dtype == "object" or pd.api.types.is_categorical_dtype(series):
            non_null = series.dropna()
            if non_null.empty:
                categorical_cols.append(col)
                continue
            try:
                parsed = pd.to_datetime(non_null, errors="coerce")
                if parsed.notna().sum() / max(len(non_null), 1) > 0.8:
                    date_cols.append(col)
                    continue
            except Exception:
                pass

            unique_ratio = non_null.nunique() / max(len(non_null), 1)
            if unique_ratio < 0.5 or non_null.nunique() <= 20:
                categorical_cols.append(col)
            else:
                text_cols.append(col)
        else:
            categorical_cols.append(col)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "date": date_cols,
        "text": text_cols,
    }


def get_dataset_overview(df: pd.DataFrame) -> Dict[str, object]:
    """Return row/column summary and column metadata."""
    dtypes = df.dtypes.astype(str).to_dict()
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "dtypes": dtypes,
    }
