from typing import Dict, Iterable, List, Optional

import pandas as pd


def detect_quality_issues(df: pd.DataFrame) -> Dict[str, object]:
    """Detect missing values, duplicate rows, type issues, and empty columns."""
    missing = df.isna().sum().sort_values(ascending=False)
    missing_df = missing[missing > 0].reset_index()
    missing_df.columns = ["column", "missing_count"]

    empty_cols = [col for col in df.columns if df[col].isna().all() or df[col].astype(str).str.strip().eq("").all()]
    duplicate_rows = int(df.duplicated().sum())

    return {
        "missing_by_column": missing_df,
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": duplicate_rows,
        "empty_columns": empty_cols,
        "column_types": df.dtypes.astype(str).to_dict(),
    }


def clean_dataset(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    fill_missing: str = "mode",
    remove_columns: Optional[Iterable[str]] = None,
    convert_datetime_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Apply selected cleaning steps and return a cleaned dataset."""
    cleaned = df.copy()

    if remove_columns:
        cols_to_drop = [col for col in remove_columns if col in cleaned.columns]
        cleaned = cleaned.drop(columns=cols_to_drop, errors="ignore")

    if drop_duplicates:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    if convert_datetime_columns:
        for col in convert_datetime_columns:
            if col in cleaned.columns:
                try:
                    cleaned[col] = pd.to_datetime(cleaned[col], errors="coerce")
                except Exception:
                    pass

    if fill_missing and fill_missing != "none":
        for col in cleaned.columns:
            if cleaned[col].isna().sum() == 0:
                continue
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                if fill_missing == "median":
                    cleaned[col] = cleaned[col].fillna(cleaned[col].median())
                elif fill_missing == "mean":
                    cleaned[col] = cleaned[col].fillna(cleaned[col].mean())
                elif fill_missing == "zero":
                    cleaned[col] = cleaned[col].fillna(0)
            else:
                if fill_missing in {"mode", "most_frequent"}:
                    mode_value = cleaned[col].mode(dropna=True)
                    if not mode_value.empty:
                        cleaned[col] = cleaned[col].fillna(mode_value.iloc[0])
                elif fill_missing == "empty_string":
                    cleaned[col] = cleaned[col].fillna("")
                elif fill_missing == "zero":
                    cleaned[col] = cleaned[col].fillna(0)

    cleaned = cleaned.drop(columns=[col for col in cleaned.columns if cleaned[col].isna().all() or cleaned[col].astype(str).str.strip().eq("").all()], errors="ignore")
    return cleaned.reset_index(drop=True)
