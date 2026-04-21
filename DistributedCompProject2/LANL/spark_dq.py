"""
spark_dq.py
===========

Purpose
-------
Reusable Spark data-quality utilities.

Why this matters
----------------
A senior engineer does not duplicate DQ logic in every script.

What we check
-------------
- missing values
- duplicate full rows
- duplicate business keys
- time horizon
- numeric distributions
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def write_json_report(path: Path, payload: dict[str, Any]) -> None:
    """
    Write a JSON report to disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def basic_overview(df: DataFrame, label: str) -> dict[str, Any]:
    """
    Row count, column names, schema shape.
    """
    return {
        "label": label,
        "row_count": int(df.count()),
        "column_count": len(df.columns),
        "columns": list(df.columns),
    }


def missing_value_report(df: DataFrame) -> dict[str, Any]:
    """
    Count nulls by column and convert to percentages.
    """
    total_rows = df.count()

    null_exprs = [
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in df.columns
    ]

    row = df.select(null_exprs).collect()[0].asDict()

    pct = {
        col: (count / total_rows * 100.0 if total_rows > 0 else 0.0)
        for col, count in row.items()
    }

    return {
        "total_rows": int(total_rows),
        "missing_count_by_column": {k: int(v) for k, v in row.items()},
        "missing_pct_by_column": pct,
    }


def full_row_duplicate_report(df: DataFrame) -> dict[str, Any]:
    """
    Full-row duplicate check.
    """
    total_rows = df.count()
    distinct_rows = df.dropDuplicates().count()
    duplicate_rows = total_rows - distinct_rows

    return {
        "total_rows": int(total_rows),
        "distinct_rows": int(distinct_rows),
        "duplicate_rows": int(duplicate_rows),
        "duplicate_pct": (duplicate_rows / total_rows * 100.0 if total_rows > 0 else 0.0),
    }


def business_key_duplicate_report(df: DataFrame, key_cols: list[str]) -> dict[str, Any]:
    """
    Check duplicates on the business key rather than the entire row.
    """
    dupes = (
        df.groupBy(*key_cols)
        .count()
        .filter(F.col("count") > 1)
    )

    duplicate_key_rows = dupes.count()

    return {
        "key_columns": key_cols,
        "duplicate_key_rows": int(duplicate_key_rows),
        "sample_duplicate_keys": [r.asDict() for r in dupes.limit(10).collect()],
    }


def time_horizon_report(df: DataFrame, time_col: str = "time") -> dict[str, Any]:
    """
    Report min/max/spread of event time.
    """
    stats = df.select(
        F.min(F.col(time_col)).alias("min_time"),
        F.max(F.col(time_col)).alias("max_time"),
    ).collect()[0]

    min_time = stats["min_time"]
    max_time = stats["max_time"]

    if min_time is None or max_time is None:
        return {
            "time_col": time_col,
            "min_time": None,
            "max_time": None,
            "time_span_seconds": None,
        }

    return {
        "time_col": time_col,
        "min_time": int(min_time),
        "max_time": int(max_time),
        "time_span_seconds": int(max_time - min_time),
    }


def numeric_distribution_report(df: DataFrame, cols: list[str]) -> dict[str, Any]:
    """
    Basic numeric shape checks.
    We use approxQuantile for percentiles because it scales better.
    """
    report: dict[str, Any] = {}

    for col_name in cols:
        non_null_count = df.filter(F.col(col_name).isNotNull()).count()

        if non_null_count == 0:
            report[col_name] = {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "q01": None,
                "q25": None,
                "median": None,
                "q75": None,
                "q99": None,
                "max": None,
            }
            continue

        stats = df.select(
            F.mean(F.col(col_name)).alias("mean"),
            F.stddev(F.col(col_name)).alias("std"),
            F.min(F.col(col_name)).alias("min"),
            F.max(F.col(col_name)).alias("max"),
        ).collect()[0]

        q01, q25, q50, q75, q99 = df.approxQuantile(col_name, [0.01, 0.25, 0.50, 0.75, 0.99], 0.001)

        report[col_name] = {
            "count": int(non_null_count),
            "mean": float(stats["mean"]) if stats["mean"] is not None else None,
            "std": float(stats["std"]) if stats["std"] is not None else 0.0,
            "min": float(stats["min"]) if stats["min"] is not None else None,
            "q01": float(q01),
            "q25": float(q25),
            "median": float(q50),
            "q75": float(q75),
            "q99": float(q99),
            "max": float(stats["max"]) if stats["max"] is not None else None,
        }

    return report