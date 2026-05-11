"""
jobs/bronze_to_silver_emr.py
============================

EMR Spark job:
Bronze (raw S3 files) -> Silver (cleaned typed parquet tables in S3)

Purpose
-------
This job reads each raw LANL source from S3, applies explicit schema,
performs light source-specific cleaning, adds partition helpers, and writes
clean silver outputs back to S3.

Why silver exists
-----------------
Silver preserves row grain while making the data usable:
- correct data types
- standardized casing / whitespace
- basic helper flags
- consistent paths and partitions

This is NOT the aggregation layer.
That happens in gold.
"""

from __future__ import annotations

import logging
import sys
from typing import Callable

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from config.settings import settings
from config.schemas import SOURCE_SCHEMAS, SOURCE_METADATA


# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# SPARK
# ============================================================
def build_spark_session(app_name: str) -> SparkSession:
    """
    Build the Spark session for EMR.

    Step by step:
    1. Start SparkSession builder
    2. Apply adaptive query execution if enabled
    3. Set shuffle partitions for large groupBy/join workloads
    4. Return the Spark session
    """
    builder = SparkSession.builder.appName(app_name)

    builder = builder.config("spark.sql.shuffle.partitions", str(settings.shuffle_partitions))
    builder = builder.config("spark.sql.adaptive.enabled", str(settings.adaptive_enabled).lower())

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ============================================================
# HELPERS
# ============================================================
def log_df_preview(df: DataFrame, label: str, n: int = 5) -> None:
    """
    Print a compact preview of the dataframe for debugging.
    """
    logger.info("=" * 80)
    logger.info(f"PREVIEW: {label}")
    df.show(n, truncate=False)


def trim_string_columns(df: DataFrame) -> DataFrame:
    """
    Strip leading/trailing whitespace from every string column.

    This is a safe silver-layer cleanup step.
    """
    for field in df.schema.fields:
        if field.dataType.simpleString() == "string":
            df = df.withColumn(field.name, F.trim(F.col(field.name)))
    return df


def add_silver_partition_helper(df: DataFrame, time_col: str = "time") -> DataFrame:
    """
    Add a day-scale event bucket for partitioning.

    Why:
    The LANL data uses relative seconds. Partitioning by a day-style bucket
    makes downstream reads more efficient without pretending we have real dates.
    """
    return df.withColumn(
        "event_day_bucket",
        F.floor(F.col(time_col) / F.lit(settings.silver_partition_seconds)).cast("bigint")
    )


def read_raw_source(spark: SparkSession, source_name: str, path: str) -> DataFrame:
    """
    Read one raw source from S3 using an explicit schema.

    No inference is used because explicit schemas are:
    - faster
    - safer
    - more reproducible
    """
    logger.info("=" * 80)
    logger.info(f"READING BRONZE SOURCE: {source_name.upper()}")
    logger.info(f"PATH: {path}")
    logger.info(f"ROW GRAIN: {SOURCE_METADATA[source_name]['row_grain']}")

    df = (
        spark.read
        .option("header", "false")
        .option("sep", ",")
        .schema(SOURCE_SCHEMAS[source_name])
        .csv(path)
    )

    return df


# ============================================================
# SOURCE-SPECIFIC CLEANING
# ============================================================
def base_clean(df: DataFrame, source_name: str) -> DataFrame:
    """
    Source-agnostic cleaning:
    1. trim string columns
    2. cast time to bigint
    3. add event_day_bucket for partitioning
    """
    time_col = SOURCE_METADATA[source_name]["time_col"]

    df = trim_string_columns(df)
    df = df.withColumn(time_col, F.col(time_col).cast("bigint"))
    df = add_silver_partition_helper(df, time_col=time_col)

    return df


def clean_auth(df: DataFrame) -> DataFrame:
    """
    Auth-specific silver cleaning.

    Adds:
    - success_flag
    """
    df = base_clean(df, "auth")

    df = df.withColumn(
        "success_flag",
        F.when(F.upper(F.col("success")) == "SUCCESS", 1).otherwise(0)
    )

    return df


def clean_flows(df: DataFrame) -> DataFrame:
    """
    Flow-specific silver cleaning.

    Important:
    We leave src_port and dst_port as strings because symbolic port-like values
    may appear and we do not want to corrupt those.
    """
    df = base_clean(df, "flows")

    df = (
        df.withColumn("duration", F.col("duration").cast("bigint"))
          .withColumn("protocol", F.col("protocol").cast("int"))
          .withColumn("packet_count", F.col("packet_count").cast("bigint"))
          .withColumn("byte_count", F.col("byte_count").cast("bigint"))
    )

    return df


def clean_dns(df: DataFrame) -> DataFrame:
    """
    DNS-specific silver cleaning.
    """
    return base_clean(df, "dns")


def clean_proc(df: DataFrame) -> DataFrame:
    """
    Proc-specific silver cleaning.

    Standardizes START/END casing.
    """
    df = base_clean(df, "proc")
    df = df.withColumn("event_type", F.upper(F.col("event_type")))
    return df


def clean_redteam(df: DataFrame) -> DataFrame:
    """
    Redteam-specific silver cleaning.

    Adds:
    - redteam_flag = 1

    This is useful later for aggregation, but must not be used as a same-window
    target feature in model training.
    """
    df = base_clean(df, "redteam")
    df = df.withColumn("redteam_flag", F.lit(1))
    return df


CLEANERS: dict[str, Callable[[DataFrame], DataFrame]] = {
    "auth": clean_auth,
    "flows": clean_flows,
    "dns": clean_dns,
    "proc": clean_proc,
    "redteam": clean_redteam,
}


# ============================================================
# LIGHTWEIGHT DATA QUALITY
# ============================================================
def log_basic_quality(df: DataFrame, source_name: str) -> None:
    """
    Run only lightweight quality checks in the silver job.

    We intentionally avoid the heaviest duplicate audits here because
    they can add huge shuffle cost on giant sources like auth.
    """
    time_col = SOURCE_METADATA[source_name]["time_col"]

    row_count = df.count()

    time_stats = df.select(
        F.min(F.col(time_col)).alias("min_time"),
        F.max(F.col(time_col)).alias("max_time"),
    ).collect()[0]

    null_counts = df.select([
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in df.columns
    ]).collect()[0].asDict()

    logger.info(f"{source_name.upper()} ROW COUNT: {row_count:,}")
    logger.info(
        f"{source_name.upper()} TIME RANGE: "
        f"min={time_stats['min_time']} max={time_stats['max_time']}"
    )
    logger.info(f"{source_name.upper()} NULL COUNTS: {null_counts}")


# ============================================================
# WRITER
# ============================================================
def write_silver(df: DataFrame, output_path: str, source_name: str) -> None:
    """
    Write one silver dataset to S3 in parquet format.

    Partitioning by event_day_bucket makes later gold reads more efficient.
    """
    logger.info(f"WRITING SILVER {source_name.upper()} -> {output_path}")

    (
        df.write
        .mode("overwrite")
        .partitionBy("event_day_bucket")
        .parquet(output_path)
    )


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    spark = build_spark_session("lanl_bronze_to_silver_emr")

    source_plan = {
        "auth": (settings.raw_auth_path, settings.silver_auth_path),
        "flows": (settings.raw_flows_path, settings.silver_flows_path),
        "dns": (settings.raw_dns_path, settings.silver_dns_path),
        "proc": (settings.raw_proc_path, settings.silver_proc_path),
        "redteam": (settings.raw_redteam_path, settings.silver_redteam_path),
    }

    for source_name, (raw_path, silver_path) in source_plan.items():
        raw_df = read_raw_source(spark, source_name, raw_path)
        cleaned_df = CLEANERS[source_name](raw_df)

        log_df_preview(cleaned_df, f"{source_name}_silver")
        log_basic_quality(cleaned_df, source_name)
        write_silver(cleaned_df, silver_path, source_name)

    logger.info("BRONZE -> SILVER JOB COMPLETE")
    spark.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("bronze_to_silver_emr failed")
        sys.exit(1)