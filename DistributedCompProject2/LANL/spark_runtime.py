"""
spark_runtime.py
================

Purpose
-------
Create a SparkSession configured for local PySpark development.

Why this matters
----------------
A senior engineer centralizes runtime/session setup.
That way, every script gets the same Spark behavior.

Notes
-----
- This is LOCAL Spark for debugging and development.
- The same business logic can be moved into Databricks later.
- Delta support is required because the project should use Delta storage.
"""

from __future__ import annotations

import logging
import os
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


def build_spark_session(app_name: str) -> SparkSession:
    """
    Build a local Spark session with Delta support.

    Step by step:
    1. Start from a SparkSession builder
    2. Use all local cores with local[*]
    3. Enable Delta Lake extensions
    4. Create the session
    5. Set log level for readability
    """
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )

    spark = builder.getOrCreate()

    # Reduce Spark noise while keeping warnings visible
    spark.sparkContext.setLogLevel("WARN")

    logger.info("Spark session created successfully")
    logger.info(f"Spark version: {spark.version}")

    return spark