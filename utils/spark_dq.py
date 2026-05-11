"""
utils/spark_dq.py
=================

Spark-native data quality helpers for the LANL project.

Purpose
-------
This module centralizes reusable data-quality logic so it does not get
copied into every job script.

Why this matters
----------------
A senior data engineer does not duplicate:
- row counts
- missingness checks
- duplicate checks
- target balance checks
- numeric profile checks

across multiple jobs.

These utilities are intentionally Spark-native so they scale in EMR.
"""

from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def basic_overview(df: DataFrame, label: str) -> dict[str, Any]:
    """
    Return a simple structural overview of the dataframe.

    Includes:
    - label
    - row count
    - column count
    - column names
    """
    return {
        "label": label,
        "row_count": int(df.count()),
        "column_count": len(df.columns),
        "columns": list(df.columns),
    }


def missing_value_report(df: DataFrame) -> dict[str, Any]:
    """
    Count nulls by column and convert them to percentages.

    Why:
    Missingness can reveal:
    - parsing problems
    - source sparsity
    - feature reliability issues
    """
    total_rows = df.count()

    row = df.select([
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in df.columns
    ]).collect()[0].asDict()

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
    Check for full-row duplicates.

    Warning:
    This can be expensive on very large datasets because it forces a heavy
    shuffle. Use carefully on huge sources.
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
    Check duplicates on a business key rather than the full row.

    This is often more meaningful than full-row duplicates because repeated
    rows can be legitimate event repetition, while duplicate business keys
    may indicate modeling-table issues.
    """
    dupes = (
        df.groupBy(*key_cols)
        .count()
        .filter(F.col("count") > 1)
    )

    return {
        "key_columns": key_cols,
        "duplicate_key_rows": int(dupes.count()),
        "sample_duplicate_keys": [r.asDict() for r in dupes.limit(10).collect()],
    }


def sample_duplicate_report(
    df: DataFrame,
    key_cols: list[str],
    fraction: float = 0.001,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Approximate duplicate check using a random sample.

    Why this exists:
    Full duplicate checks on giant sources like auth can be very expensive.
    This gives a lightweight directional signal for development.
    """
    sample_df = df.sample(withReplacement=False, fraction=fraction, seed=seed)
    sample_rows = sample_df.count()

    sample_dupes = (
        sample_df.groupBy(*key_cols)
        .count()
        .filter(F.col("count") > 1)
    )

    return {
        "sample_fraction": fraction,
        "sample_rows": int(sample_rows),
        "sample_duplicate_key_rows": int(sample_dupes.count()),
        "sample_duplicate_keys": [r.asDict() for r in sample_dupes.limit(10).collect()],
        "note": "Sample-based only. Use full duplicate checks for final audits when compute budget allows.",
    }


def time_horizon_report(df: DataFrame, time_col: str = "time") -> dict[str, Any]:
    """
    Summarize the time horizon of a dataframe.

    Important:
    For LANL, time is relative event time, not real timestamps.
    """
    row = df.select(
        F.min(F.col(time_col)).alias("min_time"),
        F.max(F.col(time_col)).alias("max_time"),
    ).collect()[0]

    min_time = row["min_time"]
    max_time = row["max_time"]

    return {
        "time_col": time_col,
        "min_time": int(min_time) if min_time is not None else None,
        "max_time": int(max_time) if max_time is not None else None,
        "time_span_seconds": int(max_time - min_time) if min_time is not None and max_time is not None else None,
    }


def numeric_distribution_report(df: DataFrame, cols: list[str]) -> dict[str, Any]:
    """
    Produce compact distribution summaries for numeric columns.

    Why:
    Means alone hide skew, tails, and extreme outliers.
    """
    out: dict[str, Any] = {}

    for c in cols:
        non_null_count = df.filter(F.col(c).isNotNull()).count()

        if non_null_count == 0:
            out[c] = {
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
            F.mean(F.col(c)).alias("mean"),
            F.stddev(F.col(c)).alias("std"),
            F.min(F.col(c)).alias("min"),
            F.max(F.col(c)).alias("max"),
        ).collect()[0]

        q01, q25, q50, q75, q99 = df.approxQuantile(c, [0.01, 0.25, 0.50, 0.75, 0.99], 0.001)

        out[c] = {
            "count": int(non_null_count),
            "mean": float(stats["mean"]) if stats["mean"] is not None else None,
            "std": float(stats["std"]) if stats["std"] is not None else None,
            "min": float(stats["min"]) if stats["min"] is not None else None,
            "q01": float(q01),
            "q25": float(q25),
            "median": float(q50),
            "q75": float(q75),
            "q99": float(q99),
            "max": float(stats["max"]) if stats["max"] is not None else None,
        }

    return out


def class_balance_report(df: DataFrame, target_col: str) -> dict[str, Any]:
    """
    Report class balance for a binary or categorical target.

    This is essential for imbalanced classification problems.
    """
    total_rows = df.count()

    counts = {
        str(r[target_col]): int(r["count"])
        for r in df.groupBy(target_col).count().collect()
    }

    pct = {
        k: (v / total_rows * 100.0 if total_rows > 0 else 0.0)
        for k, v in counts.items()
    }

    return {
        "target_col": target_col,
        "counts": counts,
        "proportions_pct": pct,
    }