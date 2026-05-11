"""
jobs/silver_to_gold_emr.py
==========================

EMR Spark job:
Silver -> Gold

Purpose
-------
This job reads the cleaned silver tables, builds event-time windows,
aggregates source-specific behavioral features, merges them into one
computer-time-window table, and creates a leakage-safe future target.

One row in gold
---------------
One row represents:
    one computer
    in one event-time window

Target
------
target_redteam_next_window

Why this matters
----------------
We want to predict whether a computer will show redteam activity in the
NEXT window, using information available in the CURRENT window.

That avoids same-window target leakage.
"""

from __future__ import annotations

import logging
import sys
from typing import Iterable

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from config.settings import settings
from config.schemas import SOURCE_METADATA


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
    builder = SparkSession.builder.appName(app_name)
    builder = builder.config("spark.sql.shuffle.partitions", str(settings.shuffle_partitions))
    builder = builder.config("spark.sql.adaptive.enabled", str(settings.adaptive_enabled).lower())

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ============================================================
# HELPERS
# ============================================================
def read_silver(spark: SparkSession, source_name: str, path: str) -> DataFrame:
    """
    Read one silver parquet dataset from S3.
    """
    logger.info("=" * 80)
    logger.info(f"READING SILVER SOURCE: {source_name.upper()}")
    logger.info(f"PATH: {path}")

    df = spark.read.parquet(path)
    return df


def add_time_window(df: DataFrame, time_col: str = "time") -> DataFrame:
    """
    Convert relative event seconds into integer windows.

    Example with 3600-second windows:
    - seconds 0 to 3599 -> window 0
    - seconds 3600 to 7199 -> window 1
    """
    return df.withColumn(
        "time_window",
        F.floor(F.col(time_col) / F.lit(settings.window_size_seconds)).cast("bigint")
    )


def add_gold_partition_helper(df: DataFrame) -> DataFrame:
    """
    Add a day-like partition helper based on time windows.

    If time_window is hourly, dividing by 24 gives a day-ish bucket.
    """
    return df.withColumn(
        "window_day_bucket",
        F.floor(F.col("time_window") / F.lit(settings.gold_windows_per_partition_day)).cast("bigint")
    )


def log_source_window_coverage(name: str, df: DataFrame) -> None:
    """
    Log time-window coverage for each source so we can verify overlap.
    """
    stats = df.select(
        F.min("time_window").alias("min_time_window"),
        F.max("time_window").alias("max_time_window"),
        F.countDistinct("time_window").alias("n_time_windows"),
    ).collect()[0]

    logger.info(
        f"{name.upper()} COVERAGE | "
        f"min_tw={stats['min_time_window']} "
        f"max_tw={stats['max_time_window']} "
        f"n_tw={stats['n_time_windows']}"
    )


def union_keyspace(frames: Iterable[DataFrame]) -> DataFrame:
    """
    Union all (computer, time_window) keys observed across feature frames.

    This ensures we do not lose a computer-window that only appears in one source.
    """
    frames = list(frames)

    if not frames:
        raise ValueError("No feature frames provided to union_keyspace().")

    out = frames[0].select("computer", "time_window")
    for frame in frames[1:]:
        out = out.unionByName(frame.select("computer", "time_window"))

    return out.dropDuplicates(["computer", "time_window"])


def safe_ratio(numerator_col: str, denominator_col: str) -> F.Column:
    """
    Avoid divide-by-zero while computing ratios.
    """
    return (
        F.when(F.col(denominator_col) == 0, F.lit(0.0))
         .otherwise(F.col(numerator_col) / F.col(denominator_col))
    )


# ============================================================
# SOURCE FEATURE BUILDERS
# ============================================================
def build_auth_features(auth: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Build auth features from both source and destination perspectives.
    """
    auth = auth.withColumn("failure_flag", 1 - F.col("success_flag"))

    auth_src = (
        auth.groupBy("src_computer", "time_window")
        .agg(
            F.count("*").alias("auth_src_event_count"),
            F.sum("success_flag").alias("auth_src_success_count"),
            F.sum("failure_flag").alias("auth_src_failure_count"),
            F.countDistinct("dst_computer").alias("auth_src_unique_dst_computers"),
            F.countDistinct("dst_user_domain").alias("auth_src_unique_dst_users"),
        )
        .withColumnRenamed("src_computer", "computer")
    )

    auth_dst = (
        auth.groupBy("dst_computer", "time_window")
        .agg(
            F.count("*").alias("auth_dst_event_count"),
            F.sum("success_flag").alias("auth_dst_success_count"),
            F.sum("failure_flag").alias("auth_dst_failure_count"),
            F.countDistinct("src_computer").alias("auth_dst_unique_src_computers"),
            F.countDistinct("src_user_domain").alias("auth_dst_unique_src_users"),
        )
        .withColumnRenamed("dst_computer", "computer")
    )

    return auth_src, auth_dst


def build_flow_features(flows: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Build flow features from both outgoing and incoming perspectives.
    """
    flows_src = (
        flows.groupBy("src_computer", "time_window")
        .agg(
            F.count("*").alias("flows_out_count"),
            F.countDistinct("dst_computer").alias("flows_out_unique_dst_computers"),
            F.countDistinct("dst_port").alias("flows_out_unique_dst_ports"),
            F.sum("duration").alias("flows_out_total_duration"),
            F.sum("packet_count").alias("flows_out_total_packets"),
            F.sum("byte_count").alias("flows_out_total_bytes"),
            F.avg("packet_count").alias("flows_out_mean_packets"),
            F.avg("byte_count").alias("flows_out_mean_bytes"),
        )
        .withColumnRenamed("src_computer", "computer")
    )

    flows_dst = (
        flows.groupBy("dst_computer", "time_window")
        .agg(
            F.count("*").alias("flows_in_count"),
            F.countDistinct("src_computer").alias("flows_in_unique_src_computers"),
            F.countDistinct("src_port").alias("flows_in_unique_src_ports"),
            F.sum("duration").alias("flows_in_total_duration"),
            F.sum("packet_count").alias("flows_in_total_packets"),
            F.sum("byte_count").alias("flows_in_total_bytes"),
            F.avg("packet_count").alias("flows_in_mean_packets"),
            F.avg("byte_count").alias("flows_in_mean_bytes"),
        )
        .withColumnRenamed("dst_computer", "computer")
    )

    return flows_src, flows_dst


def build_dns_features(dns: DataFrame) -> DataFrame:
    """
    Build DNS lookup behavior features.
    """
    return (
        dns.groupBy("src_computer", "time_window")
        .agg(
            F.count("*").alias("dns_lookup_count"),
            F.countDistinct("resolved_computer").alias("dns_unique_resolved_computers"),
        )
        .withColumnRenamed("src_computer", "computer")
    )


def build_proc_features(proc: DataFrame) -> DataFrame:
    """
    Build process behavior features.

    START/END imbalance can be a useful signal of unusual lifecycle behavior.
    """
    proc = (
        proc.withColumn("proc_start_flag", F.when(F.col("event_type") == "START", 1).otherwise(0))
            .withColumn("proc_end_flag", F.when(F.col("event_type") == "END", 1).otherwise(0))
    )

    proc_features = (
        proc.groupBy("computer", "time_window")
        .agg(
            F.count("*").alias("proc_event_count"),
            F.sum("proc_start_flag").alias("proc_start_count"),
            F.sum("proc_end_flag").alias("proc_end_count"),
            F.countDistinct("user_domain").alias("proc_unique_users"),
            F.countDistinct("process_name").alias("proc_unique_processes"),
        )
        .withColumn("proc_start_end_imbalance", F.col("proc_start_count") - F.col("proc_end_count"))
    )

    return proc_features


def build_redteam_features(redteam: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Build current-window redteam counts.

    These are operationally useful but must be treated as leakage-sensitive
    during model training.
    """
    red_src = (
        redteam.groupBy("src_computer", "time_window")
        .agg(F.count("*").alias("redteam_src_event_count"))
        .withColumnRenamed("src_computer", "computer")
    )

    red_dst = (
        redteam.groupBy("dst_computer", "time_window")
        .agg(F.count("*").alias("redteam_dst_event_count"))
        .withColumnRenamed("dst_computer", "computer")
    )

    return red_src, red_dst


# ============================================================
# LIGHTWEIGHT GOLD QUALITY CHECKS
# ============================================================
def log_gold_quality(gold: DataFrame) -> None:
    """
    Log the most important gold-level checks.

    Heavy full duplicate audits can be added later if needed, but here we focus on:
    - row count
    - business-key uniqueness
    - target balance
    """
    row_count = gold.count()

    key_dupes = (
        gold.groupBy("computer", "time_window")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )

    target_balance = {
        str(r[settings.target_column]): int(r["count"])
        for r in gold.groupBy(settings.target_column).count().collect()
    }

    logger.info("=" * 80)
    logger.info(f"GOLD ROW COUNT: {row_count:,}")
    logger.info(f"GOLD DUPLICATE BUSINESS KEYS: {key_dupes}")
    logger.info(f"GOLD TARGET BALANCE: {target_balance}")


# ============================================================
# WRITER
# ============================================================
def write_gold(gold: DataFrame, output_path: str) -> None:
    """
    Write the gold feature table to S3 in parquet format.
    """
    logger.info(f"WRITING GOLD -> {output_path}")

    (
        gold.write
        .mode("overwrite")
        .partitionBy("window_day_bucket")
        .parquet(output_path)
    )


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    spark = build_spark_session("lanl_silver_to_gold_emr")

    # --------------------------------------------------------
    # READ SILVER
    # --------------------------------------------------------
    auth = add_time_window(read_silver(spark, "auth", settings.silver_auth_path))
    flows = add_time_window(read_silver(spark, "flows", settings.silver_flows_path))
    dns = add_time_window(read_silver(spark, "dns", settings.silver_dns_path))
    proc = add_time_window(read_silver(spark, "proc", settings.silver_proc_path))
    redteam = add_time_window(read_silver(spark, "redteam", settings.silver_redteam_path))

    # --------------------------------------------------------
    # LOG SOURCE COVERAGE
    # --------------------------------------------------------
    for source_name, df in [
        ("auth", auth),
        ("flows", flows),
        ("dns", dns),
        ("proc", proc),
        ("redteam", redteam),
    ]:
        log_source_window_coverage(source_name, df)

    # --------------------------------------------------------
    # BUILD FEATURE FRAMES
    # --------------------------------------------------------
    auth_src, auth_dst = build_auth_features(auth)
    flows_src, flows_dst = build_flow_features(flows)
    dns_feat = build_dns_features(dns)
    proc_feat = build_proc_features(proc)
    red_src, red_dst = build_redteam_features(redteam)

    # --------------------------------------------------------
    # BUILD MASTER KEYSPACE
    # --------------------------------------------------------
    keyspace = union_keyspace([
        auth_src,
        auth_dst,
        flows_src,
        flows_dst,
        dns_feat,
        proc_feat,
        red_src,
        red_dst,
    ])

    logger.info(f"KEYSPACE ROWS: {keyspace.count():,}")

    # --------------------------------------------------------
    # MERGE ALL FEATURES
    # --------------------------------------------------------
    gold = keyspace

    for feat_df in [
        auth_src,
        auth_dst,
        flows_src,
        flows_dst,
        dns_feat,
        proc_feat,
        red_src,
        red_dst,
    ]:
        gold = gold.join(feat_df, on=["computer", "time_window"], how="left")

    # Null numeric values after joins usually mean "no events in that source for this window"
    fill_cols = [c for c in gold.columns if c not in {"computer", "time_window"}]
    gold = gold.fillna(0, subset=fill_cols)

    # --------------------------------------------------------
    # DERIVED FEATURES
    # --------------------------------------------------------
    gold = (
        gold
        .withColumn("auth_total_events", F.col("auth_src_event_count") + F.col("auth_dst_event_count"))
        .withColumn("auth_total_failures", F.col("auth_src_failure_count") + F.col("auth_dst_failure_count"))
        .withColumn("auth_total_successes", F.col("auth_src_success_count") + F.col("auth_dst_success_count"))
        .withColumn("flows_total_events", F.col("flows_out_count") + F.col("flows_in_count"))
        .withColumn("flows_total_bytes", F.col("flows_out_total_bytes") + F.col("flows_in_total_bytes"))
        .withColumn("flows_total_packets", F.col("flows_out_total_packets") + F.col("flows_in_total_packets"))
        .withColumn("redteam_event_count", F.col("redteam_src_event_count") + F.col("redteam_dst_event_count"))
        .withColumn("redteam_current_flag", F.when(F.col("redteam_event_count") > 0, 1).otherwise(0))
        .withColumn("auth_failure_ratio", safe_ratio("auth_total_failures", "auth_total_events"))
        .withColumn("flows_bytes_per_event", safe_ratio("flows_total_bytes", "flows_total_events"))
        .withColumn("flows_packets_per_event", safe_ratio("flows_total_packets", "flows_total_events"))
    )

    # --------------------------------------------------------
    # FUTURE TARGET
    # --------------------------------------------------------
    # This is the critical anti-leakage choice.
    # We predict the next window's redteam status, not the current window.
    window_spec = Window.partitionBy("computer").orderBy("time_window")

    gold = gold.withColumn(
        settings.target_column,
        F.coalesce(F.lead("redteam_current_flag", 1).over(window_spec), F.lit(0))
    )

    # --------------------------------------------------------
    # PARTITION HELPER
    # --------------------------------------------------------
    gold = add_gold_partition_helper(gold)

    # --------------------------------------------------------
    # QUALITY CHECKS
    # --------------------------------------------------------
    log_gold_quality(gold)

    # --------------------------------------------------------
    # WRITE GOLD
    # --------------------------------------------------------
    write_gold(gold, settings.gold_computer_time_path)

    logger.info("SILVER -> GOLD JOB COMPLETE")
    spark.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("silver_to_gold_emr failed")
        sys.exit(1)