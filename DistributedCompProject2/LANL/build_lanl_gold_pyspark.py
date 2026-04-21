"""
build_lanl_gold_pyspark.py
==========================

Purpose
-------
Read SILVER Delta tables, build event-time windows, aggregate cross-source features,
construct a leakage-safe future target, run DQ checks, and write GOLD Delta output.

Why this matters
----------------
This is the authoritative GOLD build for the project.

One row represents
------------------
One COMPUTER in one TIME WINDOW.

Target
------
We do NOT predict same-window redteam activity using same-window redteam counts.
That would leak the answer.

Instead:
- redteam_current_flag = diagnostics / operational context only
- target_redteam_next_window = what we predict

This is a much more mature setup because it reflects what the model would know
before the next window occurs.
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, IntegerType

from lanl_contracts import (
    SILVER_OUTPUTS,
    GOLD_OUTPUT,
    SOURCE_METADATA,
    WINDOW_SIZE_SECONDS,
    TARGET_COLUMN,
    DQ_DIR,
)
from spark_runtime import build_spark_session
from spark_dq import (
    write_json_report,
    basic_overview,
    missing_value_report,
    full_row_duplicate_report,
    business_key_duplicate_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def read_silver_delta(spark, source_name: str) -> DataFrame:
    """
    Read one silver Delta source.
    """
    path = str(SILVER_OUTPUTS[source_name])
    logger.info(f"Reading silver Delta: source={source_name} path={path}")
    return spark.read.format("delta").load(path)


def add_time_window(df: DataFrame, time_col: str = "time") -> DataFrame:
    """
    Convert raw event time into fixed event-time windows.
    """
    return df.withColumn(
        "time_window",
        F.floor(F.col(time_col) / F.lit(WINDOW_SIZE_SECONDS)).cast("bigint")
    )


def build_source_time_overlap_report(source_frames: dict[str, DataFrame]) -> dict:
    """
    Report min/max window coverage for each source.
    This matters because a sample can accidentally cover different time ranges
    across sources, which weakens modeling quality.
    """
    report = {}

    for name, df in source_frames.items():
        stats = df.select(
            F.min("time_window").alias("min_time_window"),
            F.max("time_window").alias("max_time_window"),
            F.countDistinct("time_window").alias("n_time_windows"),
        ).collect()[0]

        report[name] = {
            "min_time_window": int(stats["min_time_window"]) if stats["min_time_window"] is not None else None,
            "max_time_window": int(stats["max_time_window"]) if stats["max_time_window"] is not None else None,
            "n_time_windows": int(stats["n_time_windows"]),
        }

    return report


def auth_features(auth: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Build source-side and destination-side auth features.
    """
    auth = auth.withColumn("failure_flag", 1 - F.col("success_flag"))

    src = (
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

    dst = (
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

    return src, dst


def flows_features(flows: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Build source-side and destination-side flow features.
    """
    src = (
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

    dst = (
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

    return src, dst


def dns_features(dns: DataFrame) -> DataFrame:
    return (
        dns.groupBy("src_computer", "time_window")
        .agg(
            F.count("*").alias("dns_lookup_count"),
            F.countDistinct("resolved_computer").alias("dns_unique_resolved_computers"),
        )
        .withColumnRenamed("src_computer", "computer")
    )


def proc_features(proc: DataFrame) -> DataFrame:
    proc = (
        proc.withColumn("proc_start_flag", F.when(F.col("event_type") == "START", 1).otherwise(0))
            .withColumn("proc_end_flag", F.when(F.col("event_type") == "END", 1).otherwise(0))
    )

    out = (
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

    return out


def redteam_features(red: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Build redteam current-window counts for diagnostics and future-target construction.
    """
    src = (
        red.groupBy("src_computer", "time_window")
        .agg(F.count("*").alias("redteam_src_event_count"))
        .withColumnRenamed("src_computer", "computer")
    )

    dst = (
        red.groupBy("dst_computer", "time_window")
        .agg(F.count("*").alias("redteam_dst_event_count"))
        .withColumnRenamed("dst_computer", "computer")
    )

    return src, dst


def build_master_keyspace(frames: list[DataFrame]) -> DataFrame:
    """
    Build the set of all (computer, time_window) combinations that appear anywhere.
    """
    unioned = None

    for df in frames:
        small = df.select("computer", "time_window")
        if unioned is None:
            unioned = small
        else:
            unioned = unioned.unionByName(small)

    return unioned.dropDuplicates(["computer", "time_window"])


def safe_ratio(numerator_col: str, denominator_col: str) -> F.Column:
    """
    Avoid divide-by-zero issues.
    """
    return F.when(F.col(denominator_col) == 0, F.lit(0.0)) \
            .otherwise(F.col(numerator_col) / F.col(denominator_col))


def build_dq_report(gold: DataFrame) -> dict:
    """
    Final DQ report for the gold table.
    """
    return {
        "row_grain": SOURCE_METADATA["gold"]["row_grain"],
        "overview": basic_overview(gold, "gold"),
        "missing_values": missing_value_report(gold),
        "full_row_duplicates": full_row_duplicate_report(gold),
        "business_key_duplicates": business_key_duplicate_report(
            gold, SOURCE_METADATA["gold"]["business_key_cols"]
        ),
    }


def main() -> None:
    spark = build_spark_session("lanl_gold_pipeline")

    logger.info("LANL GOLD PIPELINE STARTED")

    # Read silver
    auth = add_time_window(read_silver_delta(spark, "auth"))
    flows = add_time_window(read_silver_delta(spark, "flows"))
    dns = add_time_window(read_silver_delta(spark, "dns"))
    proc = add_time_window(read_silver_delta(spark, "proc"))
    red = add_time_window(read_silver_delta(spark, "redteam"))

    source_frames = {
        "auth": auth,
        "flows": flows,
        "dns": dns,
        "proc": proc,
        "redteam": red,
    }

    overlap_report = build_source_time_overlap_report(source_frames)
    write_json_report(DQ_DIR / "source_time_window_overlap_report.json", overlap_report)
    logger.info(f"Source overlap report: {overlap_report}")

    # Build features
    auth_src, auth_dst = auth_features(auth)
    flows_src, flows_dst = flows_features(flows)
    dns_feat = dns_features(dns)
    proc_feat = proc_features(proc)
    red_src, red_dst = redteam_features(red)

    # Build keyspace
    keyspace = build_master_keyspace([
        auth_src, auth_dst, flows_src, flows_dst,
        dns_feat, proc_feat, red_src, red_dst
    ])

    logger.info(f"Keyspace rows: {keyspace.count()}")

    # Merge features
    gold = keyspace
    for feature_df in [auth_src, auth_dst, flows_src, flows_dst, dns_feat, proc_feat, red_src, red_dst]:
        gold = gold.join(feature_df, on=["computer", "time_window"], how="left")

    # Fill null numeric columns with zero
    numeric_cols = [
        f.name for f in gold.schema.fields
        if isinstance(f.dataType, (LongType, IntegerType))
    ]
    gold = gold.fillna(0, subset=numeric_cols)

    # Derived features
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

    # Future target
    # Predict whether the NEXT window for the same computer contains redteam activity.
    w = Window.partitionBy("computer").orderBy("time_window")

    gold = gold.withColumn(
        TARGET_COLUMN,
        F.coalesce(F.lead("redteam_current_flag", 1).over(w), F.lit(0))
    )

    # Final DQ report
    gold_dq = build_dq_report(gold)
    gold_dq["target_distribution"] = {
        str(r[TARGET_COLUMN]): int(r["count"])
        for r in gold.groupBy(TARGET_COLUMN).count().collect()
    }
    write_json_report(DQ_DIR / "gold_dq_report.json", gold_dq)

    logger.info(f"Gold target distribution: {gold_dq['target_distribution']}")

    # Write Delta
    gold_path = str(GOLD_OUTPUT)
    logger.info(f"Writing gold Delta to {gold_path}")

    (
        gold.write
        .format("delta")
        .mode("overwrite")
        .save(gold_path)
    )

    logger.info("LANL GOLD PIPELINE COMPLETE")
    spark.stop()


if __name__ == "__main__":
    main()