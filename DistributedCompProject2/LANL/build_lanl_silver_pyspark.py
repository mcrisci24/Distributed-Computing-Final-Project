"""
build_lanl_silver_pyspark.py
============================

Purpose
-------
Read raw LANL files, apply source-specific cleaning, run data-quality checks,
and write SILVER outputs in Delta format.

Why this matters
----------------
This is the authoritative SILVER build for the project.
It keeps each source separate and preserves raw event grain.

One row represents
------------------
- auth   = one authentication event
- flows  = one network flow
- dns    = one DNS resolution event
- proc   = one process lifecycle event
- redteam= one labeled compromise event

Important analytical maturity notes
-----------------------------------
- Time is relative event time, not real dates.
- This means we can reason about event ordering and windows,
  but NOT about calendar seasonality.
- Duplicates are not automatically bad in cyber data.
  Many repeated events are real behavior.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    LongType,
    IntegerType,
    StringType,
)

from lanl_contracts import (
    RAW_FILES,
    SILVER_OUTPUTS,
    SOURCE_METADATA,
    DQ_DIR,
)
from spark_runtime import build_spark_session
from spark_dq import (
    write_json_report,
    basic_overview,
    missing_value_report,
    full_row_duplicate_report,
    business_key_duplicate_report,
    time_horizon_report,
    numeric_distribution_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# SPARK SCHEMAS
# We define Spark schemas explicitly instead of relying on inference.
# This is safer, faster, and more reproducible.
# ============================================================
AUTH_SCHEMA = StructType([
    StructField("time", LongType(), True),
    StructField("src_user_domain", StringType(), True),
    StructField("dst_user_domain", StringType(), True),
    StructField("src_computer", StringType(), True),
    StructField("dst_computer", StringType(), True),
    StructField("auth_type", StringType(), True),
    StructField("logon_type", StringType(), True),
    StructField("auth_orientation", StringType(), True),
    StructField("success", StringType(), True),
])

FLOWS_SCHEMA = StructType([
    StructField("time", LongType(), True),
    StructField("duration", LongType(), True),
    StructField("src_computer", StringType(), True),
    StructField("src_port", StringType(), True),
    StructField("dst_computer", StringType(), True),
    StructField("dst_port", StringType(), True),
    StructField("protocol", IntegerType(), True),
    StructField("packet_count", LongType(), True),
    StructField("byte_count", LongType(), True),
])

DNS_SCHEMA = StructType([
    StructField("time", LongType(), True),
    StructField("src_computer", StringType(), True),
    StructField("resolved_computer", StringType(), True),
])

PROC_SCHEMA = StructType([
    StructField("time", LongType(), True),
    StructField("user_domain", StringType(), True),
    StructField("computer", StringType(), True),
    StructField("process_name", StringType(), True),
    StructField("event_type", StringType(), True),
])

REDTEAM_SCHEMA = StructType([
    StructField("time", LongType(), True),
    StructField("user_domain", StringType(), True),
    StructField("src_computer", StringType(), True),
    StructField("dst_computer", StringType(), True),
])

SCHEMAS = {
    "auth": AUTH_SCHEMA,
    "flows": FLOWS_SCHEMA,
    "dns": DNS_SCHEMA,
    "proc": PROC_SCHEMA,
    "redteam": REDTEAM_SCHEMA,
}


def trim_string_columns(df: DataFrame) -> DataFrame:
    """
    Strip whitespace from every string column.
    """
    for field in df.schema.fields:
        if isinstance(field.dataType, StringType):
            df = df.withColumn(field.name, F.trim(F.col(field.name)))
    return df


def read_raw_source(spark, source_name: str) -> DataFrame:
    """
    Read one raw source file with explicit schema.
    """
    path = str(RAW_FILES[source_name])
    logger.info(f"Reading raw source: {source_name} | path={path}")

    df = (
        spark.read
        .option("header", "false")
        .option("sep", ",")
        .schema(SCHEMAS[source_name])
        .csv(path)
    )

    return df


def basic_clean(df: DataFrame, source_name: str) -> DataFrame:
    """
    Source-agnostic cleaning:
    1. trim strings
    2. ensure time is bigint
    """
    df = trim_string_columns(df)

    time_col = SOURCE_METADATA[source_name]["time_col"]
    if time_col in df.columns:
        df = df.withColumn(time_col, F.col(time_col).cast("bigint"))

    return df


def clean_auth(df: DataFrame) -> DataFrame:
    """
    Add success_flag for auth modeling later.
    """
    df = basic_clean(df, "auth")
    df = df.withColumn(
        "success_flag",
        F.when(F.upper(F.col("success")) == "SUCCESS", F.lit(1)).otherwise(F.lit(0))
    )
    return df


def clean_flows(df: DataFrame) -> DataFrame:
    """
    Flow cleaning.
    NOTE:
    We intentionally do NOT force ports to numeric because this dataset
    can contain symbolic labels such as N10451.
    """
    df = basic_clean(df, "flows")
    df = (
        df.withColumn("duration", F.col("duration").cast("bigint"))
          .withColumn("protocol", F.col("protocol").cast("int"))
          .withColumn("packet_count", F.col("packet_count").cast("bigint"))
          .withColumn("byte_count", F.col("byte_count").cast("bigint"))
    )
    return df


def clean_dns(df: DataFrame) -> DataFrame:
    df = basic_clean(df, "dns")
    return df


def clean_proc(df: DataFrame) -> DataFrame:
    df = basic_clean(df, "proc")
    df = df.withColumn("event_type", F.upper(F.col("event_type")))
    return df


def clean_redteam(df: DataFrame) -> DataFrame:
    df = basic_clean(df, "redteam")
    df = df.withColumn("redteam_flag", F.lit(1))
    return df


CLEANERS = {
    "auth": clean_auth,
    "flows": clean_flows,
    "dns": clean_dns,
    "proc": clean_proc,
    "redteam": clean_redteam,
}


def build_dq_report(df: DataFrame, source_name: str) -> dict:
    """
    Build a full DQ report for one source.
    """
    numeric_cols = [
        f.name for f in df.schema.fields
        if isinstance(f.dataType, (LongType, IntegerType))
    ]

    report = {
        "source_name": source_name,
        "row_grain": SOURCE_METADATA[source_name]["row_grain"],
        "notes": SOURCE_METADATA[source_name]["notes"],
        "overview": basic_overview(df, source_name),
        "time_horizon": time_horizon_report(df, SOURCE_METADATA[source_name]["time_col"]),
        "missing_values": missing_value_report(df),
        "full_row_duplicates": full_row_duplicate_report(df),
        "business_key_duplicates": business_key_duplicate_report(
            df, SOURCE_METADATA[source_name]["business_key_cols"]
        ),
        "numeric_distributions": numeric_distribution_report(df, numeric_cols[:10]) if numeric_cols else {},
    }
    return report


def write_silver_delta(df: DataFrame, source_name: str) -> None:
    """
    Write cleaned source as Delta.
    """
    out_path = str(SILVER_OUTPUTS[source_name])

    logger.info(f"Writing silver Delta: source={source_name} path={out_path}")

    (
        df.write
        .format("delta")
        .mode("overwrite")
        .save(out_path)
    )


def main() -> None:
    spark = build_spark_session("lanl_silver_pipeline")

    logger.info("LANL SILVER PIPELINE STARTED")

    for source_name in ["auth", "flows", "dns", "proc", "redteam"]:
        logger.info("=" * 80)
        logger.info(f"PROCESSING SOURCE: {source_name.upper()}")

        raw_df = read_raw_source(spark, source_name)
        clean_df = CLEANERS[source_name](raw_df)

        # Log quick shape information
        logger.info(f"{source_name} row count after clean: {clean_df.count()}")
        logger.info(f"{source_name} columns: {clean_df.columns}")

        # Save DQ report
        dq_report = build_dq_report(clean_df, source_name)
        write_json_report(DQ_DIR / f"{source_name}_silver_dq_report.json", dq_report)

        # Write Delta
        write_silver_delta(clean_df, source_name)

    logger.info("LANL SILVER PIPELINE COMPLETE")
    spark.stop()


if __name__ == "__main__":
    main()