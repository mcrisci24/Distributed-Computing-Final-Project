from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def write_json_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def missing_value_report(df: pd.DataFrame) -> dict:
    total_rows = len(df)
    counts = df.isna().sum().to_dict()
    pct = {
        col: (count / total_rows * 100.0 if total_rows > 0 else 0.0)
        for col, count in counts.items()
    }
    return {
        "total_rows": total_rows,
        "missing_count_by_column": counts,
        "missing_pct_by_column": pct,
    }


def full_row_duplicate_report(df: pd.DataFrame) -> dict:
    total_rows = len(df)
    distinct_rows = len(df.drop_duplicates())
    duplicate_rows = total_rows - distinct_rows
    return {
        "total_rows": total_rows,
        "distinct_rows": distinct_rows,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": (duplicate_rows / total_rows * 100.0 if total_rows > 0 else 0.0),
    }


def business_key_duplicate_report(df: pd.DataFrame, key_cols: list[str]) -> dict:
    dupes = (
        df.groupby(key_cols, dropna=False)
        .size()
        .reset_index(name="count")
    )
    dupes = dupes[dupes["count"] > 1]
    return {
        "key_columns": key_cols,
        "duplicate_key_rows": len(dupes),
        "sample_duplicate_keys": dupes.head(10).to_dict(orient="records"),
    }


def numeric_distribution_report(df: pd.DataFrame, cols: list[str]) -> dict:
    out = {}
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        out[col] = {
            "count": int(s.count()),
            "mean": float(s.mean()) if s.count() else None,
            "std": float(s.std()) if s.count() else None,
            "min": float(s.min()) if s.count() else None,
            "q01": float(s.quantile(0.01)) if s.count() else None,
            "q25": float(s.quantile(0.25)) if s.count() else None,
            "median": float(s.median()) if s.count() else None,
            "q75": float(s.quantile(0.75)) if s.count() else None,
            "q99": float(s.quantile(0.99)) if s.count() else None,
            "max": float(s.max()) if s.count() else None,
            "skew": float(s.skew()) if s.count() else None,
        }
    return out


def time_horizon_report(df: pd.DataFrame, time_col: str = "time") -> dict:
    s = pd.to_numeric(df[time_col], errors="coerce")
    return {
        "time_col": time_col,
        "min_time": float(s.min()) if s.count() else None,
        "max_time": float(s.max()) if s.count() else None,
        "time_span_seconds": float(s.max() - s.min()) if s.count() else None,
    }