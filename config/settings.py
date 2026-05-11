"""
config/settings.py
==================

Central configuration for the LANL distributed pipeline.

Why this file exists
--------------------
A senior data engineer does not scatter:
- S3 paths
- runtime flags
- partition sizes
- model target names
- project constants

across many files.

This module is the single source of truth for pipeline settings.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    """
    Immutable runtime settings for the project.

    These values can be overridden with environment variables if needed.
    """

    # ------------------------------------------------------------
    # AWS / S3 SETTINGS
    # ------------------------------------------------------------
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    s3_bucket: str = os.getenv("LANL_S3_BUCKET", "lanl-cyber-pipeline")
    s3_prefix: str = os.getenv("LANL_S3_PREFIX", "lanl")

    # ------------------------------------------------------------
    # BRONZE / SILVER / GOLD PATHS
    # ------------------------------------------------------------
    @property
    def bronze_base(self) -> str:
        return f"s3://{self.s3_bucket}/{self.s3_prefix}/bronze"

    @property
    def silver_base(self) -> str:
        return f"s3://{self.s3_bucket}/{self.s3_prefix}/silver"

    @property
    def gold_base(self) -> str:
        return f"s3://{self.s3_bucket}/{self.s3_prefix}/gold"

    @property
    def model_base(self) -> str:
        return f"s3://{self.s3_bucket}/{self.s3_prefix}/model"

    # ------------------------------------------------------------
    # RAW SOURCE PATHS
    # ------------------------------------------------------------
    @property
    def raw_auth_path(self) -> str:
        return f"{self.bronze_base}/auth.txt.gz"

    @property
    def raw_flows_path(self) -> str:
        return f"{self.bronze_base}/flows.txt.gz"

    @property
    def raw_dns_path(self) -> str:
        return f"{self.bronze_base}/dns.txt.gz"

    @property
    def raw_proc_path(self) -> str:
        return f"{self.bronze_base}/proc.txt.gz"

    @property
    def raw_redteam_path(self) -> str:
        return f"{self.bronze_base}/redteam.txt.gz"

    # ------------------------------------------------------------
    # SILVER OUTPUT PATHS
    # ------------------------------------------------------------
    @property
    def silver_auth_path(self) -> str:
        return f"{self.silver_base}/auth"

    @property
    def silver_flows_path(self) -> str:
        return f"{self.silver_base}/flows"

    @property
    def silver_dns_path(self) -> str:
        return f"{self.silver_base}/dns"

    @property
    def silver_proc_path(self) -> str:
        return f"{self.silver_base}/proc"

    @property
    def silver_redteam_path(self) -> str:
        return f"{self.silver_base}/redteam"

    # ------------------------------------------------------------
    # GOLD OUTPUT PATH
    # ------------------------------------------------------------
    @property
    def gold_computer_time_path(self) -> str:
        return f"{self.gold_base}/computer_time"

    # ------------------------------------------------------------
    # TABLE NAMES (OPTIONAL / FOR CATALOG REGISTRATION)
    # ------------------------------------------------------------
    silver_auth_table: str = "lanl_silver_auth"
    silver_flows_table: str = "lanl_silver_flows"
    silver_dns_table: str = "lanl_silver_dns"
    silver_proc_table: str = "lanl_silver_proc"
    silver_redteam_table: str = "lanl_silver_redteam"
    gold_table: str = "lanl_gold_computer_time"

    # ------------------------------------------------------------
    # FEATURE ENGINEERING SETTINGS
    # ------------------------------------------------------------
    # Event-time windows. LANL time is relative seconds, not calendar dates.
    window_size_seconds: int = int(os.getenv("LANL_WINDOW_SIZE_SECONDS", "3600"))

    # Silver partition helper. One "day" = 86400 seconds of relative event time.
    silver_partition_seconds: int = int(os.getenv("LANL_SILVER_PARTITION_SECONDS", "86400"))

    # Gold partition helper. Since time_window is hourly, 24 windows ~= one day bucket.
    gold_windows_per_partition_day: int = int(os.getenv("LANL_GOLD_WINDOWS_PER_DAY", "24"))

    # ------------------------------------------------------------
    # TARGET / LEAKAGE SETTINGS
    # ------------------------------------------------------------
    target_column: str = "target_redteam_next_window"

    current_window_redteam_columns: tuple[str, ...] = (
        "redteam_src_event_count",
        "redteam_dst_event_count",
        "redteam_event_count",
        "redteam_current_flag",
    )

    non_feature_columns: tuple[str, ...] = (
        "computer",
        "time_window",
        "window_day_bucket",
        "target_redteam_next_window",
    )

    # ------------------------------------------------------------
    # WRITE FORMAT
    # ------------------------------------------------------------
    # Use parquet on EMR for maximum simplicity and compatibility.
    storage_format: str = os.getenv("LANL_STORAGE_FORMAT", "parquet")

    # ------------------------------------------------------------
    # SPARK TUNING
    # ------------------------------------------------------------
    shuffle_partitions: int = int(os.getenv("LANL_SHUFFLE_PARTITIONS", "200"))
    adaptive_enabled: bool = os.getenv("LANL_ADAPTIVE_ENABLED", "true").lower() == "true"

    # ------------------------------------------------------------
    # LOGGING / DEBUG
    # ------------------------------------------------------------
    log_level: str = os.getenv("LANL_LOG_LEVEL", "INFO")


settings = Settings()