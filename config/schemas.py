"""
config/schemas.py
=================

Explicit source schemas and metadata for the LANL dataset.

Why this file exists
--------------------
We do not want schema inference for a large distributed pipeline.
Inference can be:
- slow
- inconsistent
- error-prone

Explicit schemas make the pipeline:
- reproducible
- understandable
- easier to debug
"""

from __future__ import annotations

from pyspark.sql.types import (
    StructType,
    StructField,
    LongType,
    IntegerType,
    StringType,
)

# ------------------------------------------------------------
# AUTH SCHEMA
# One row = one authentication event
# ------------------------------------------------------------
AUTH_COLUMNS = [
    "time",
    "src_user_domain",
    "dst_user_domain",
    "src_computer",
    "dst_computer",
    "auth_type",
    "logon_type",
    "auth_orientation",
    "success",
]

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

# ------------------------------------------------------------
# FLOWS SCHEMA
# One row = one network flow
# ------------------------------------------------------------
FLOWS_COLUMNS = [
    "time",
    "duration",
    "src_computer",
    "src_port",
    "dst_computer",
    "dst_port",
    "protocol",
    "packet_count",
    "byte_count",
]

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

# ------------------------------------------------------------
# DNS SCHEMA
# One row = one DNS resolution event
# ------------------------------------------------------------
DNS_COLUMNS = [
    "time",
    "src_computer",
    "resolved_computer",
]

DNS_SCHEMA = StructType([
    StructField("time", LongType(), True),
    StructField("src_computer", StringType(), True),
    StructField("resolved_computer", StringType(), True),
])

# ------------------------------------------------------------
# PROC SCHEMA
# One row = one process lifecycle event
# ------------------------------------------------------------
PROC_COLUMNS = [
    "time",
    "user_domain",
    "computer",
    "process_name",
    "event_type",
]

PROC_SCHEMA = StructType([
    StructField("time", LongType(), True),
    StructField("user_domain", StringType(), True),
    StructField("computer", StringType(), True),
    StructField("process_name", StringType(), True),
    StructField("event_type", StringType(), True),
])

# REDTEAM Schema
# One row = one labeled compromise event
REDTEAM_COLUMNS = [
    "time",
    "user_domain",
    "src_computer",
    "dst_computer",
]

REDTEAM_SCHEMA = StructType([
    StructField("time", LongType(), True),
    StructField("user_domain", StringType(), True),
    StructField("src_computer", StringType(), True),
    StructField("dst_computer", StringType(), True),
])

# Registries
SOURCE_SCHEMAS = {
    "auth": AUTH_SCHEMA,
    "flows": FLOWS_SCHEMA,
    "dns": DNS_SCHEMA,
    "proc": PROC_SCHEMA,
    "redteam": REDTEAM_SCHEMA,
}

SOURCE_COLUMNS = {
    "auth": AUTH_COLUMNS,
    "flows": FLOWS_COLUMNS,
    "dns": DNS_COLUMNS,
    "proc": PROC_COLUMNS,
    "redteam": REDTEAM_COLUMNS,
}

# Source metadata
SOURCE_METADATA = {
    "auth": {
        "row_grain": "one authentication event",
        "time_col": "time",
        "business_key_cols": [
            "time", "src_user_domain", "dst_user_domain",
            "src_computer", "dst_computer", "auth_type",
            "logon_type", "auth_orientation", "success",
        ],
        "notes": [
            "Relative event time only. No calendar dates exist.",
            "Repeated auth events may be real behavior, not bad data.",
        ],
    },
    "flows": {
        "row_grain": "one network flow",
        "time_col": "time",
        "business_key_cols": [
            "time", "duration", "src_computer", "src_port",
            "dst_computer", "dst_port", "protocol",
            "packet_count", "byte_count",
        ],
        "notes": [
            "Ports may be symbolic labels in some environments, so they remain strings.",
            "Repeated flow patterns may be meaningful behavior.",
        ],
    },
    "dns": {
        "row_grain": "one DNS resolution event",
        "time_col": "time",
        "business_key_cols": ["time", "src_computer", "resolved_computer"],
        "notes": [
            "Repeated DNS lookups may be expected in normal operation.",
        ],
    },
    "proc": {
        "row_grain": "one process lifecycle event",
        "time_col": "time",
        "business_key_cols": ["time", "user_domain", "computer", "process_name", "event_type"],
        "notes": [
            "This is one row per event, not one row per process.",
        ],
    },
    "redteam": {
        "row_grain": "one labeled compromise event",
        "time_col": "time",
        "business_key_cols": ["time", "user_domain", "src_computer", "dst_computer"],
        "notes": [
            "This is the ground-truth attack label source.",
            "Current-window redteam features are useful diagnostically but are leakage risks for modeling.",
        ],
    },
    "gold": {
        "row_grain": "one computer in one event-time window",
        "business_key_cols": ["computer", "time_window"],
        "notes": [
            "Gold is the model-ready table.",
            "Prediction target should be next-window redteam activity, not current-window activity.",
        ],
    },
}