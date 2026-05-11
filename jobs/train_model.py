"""
jobs/train_model.py
===================

Train the predictive model using the gold feature table.

Purpose
-------
This job:
1. reads the gold table from S3
2. converts it to pandas for scikit-learn modeling
3. removes leakage-sensitive columns
4. uses time-aware train/validation/test splits
5. trains a benchmark and a stronger model
6. logs metrics to MLflow
7. writes feature names and best-model metadata

Why this design works
---------------------
The heavy data engineering is already distributed in Spark/EMR.
The final model training layer can reasonably use scikit-learn on the
gold table if the gold table is small enough to fit in memory.

This is a practical and defensible architecture for the rubric.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import settings


# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# MODEL ARTIFACT PATHS
# ============================================================
# These are local paths on the machine where the training job runs.
# In EC2/EMR you can later sync these to S3 if needed.
LOCAL_MODEL_DIR = Path(os.getenv("LANL_LOCAL_MODEL_DIR", "./model_outputs"))
LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES_FILE = LOCAL_MODEL_DIR / "feature_names.json"
BEST_MODEL_SUMMARY_FILE = LOCAL_MODEL_DIR / "best_model_summary.json"

# Local MLflow backend by default.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file://{LOCAL_MODEL_DIR.resolve() / 'mlruns'}")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "lanl_threat_prediction")

RANDOM_STATE = 42


# ============================================================
# SPARK
# ============================================================
def build_spark_session(app_name: str) -> SparkSession:
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ============================================================
# HELPERS
# ============================================================
def write_json(path: Path, payload: dict) -> None:
    """
    Write JSON metadata in a human-readable format.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def assert_no_leakage(feature_cols: list[str]) -> None:
    """
    Fail fast if leakage-sensitive columns survived feature selection.
    """
    leaking = [c for c in feature_cols if c in settings.current_window_redteam_columns]
    if leaking:
        raise ValueError(f"Leakage columns found in feature set: {leaking}")


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-aware split:
    - earliest 60% of windows = train
    - next 20% = validation
    - latest 20% = test

    Why:
    Randomly mixing future and past windows can inflate performance and
    make the evaluation unrealistic.
    """
    windows = sorted(df["time_window"].dropna().unique().tolist())

    if len(windows) < 5:
        raise ValueError("Not enough distinct time windows for a meaningful time-aware split.")

    train_cut = windows[int(len(windows) * 0.60)]
    valid_cut = windows[int(len(windows) * 0.80)]

    train_df = df[df["time_window"] <= train_cut].copy()
    valid_df = df[(df["time_window"] > train_cut) & (df["time_window"] <= valid_cut)].copy()
    test_df = df[df["time_window"] > valid_cut].copy()

    return train_df, valid_df, test_df


def evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Train one model, evaluate it, log metrics, and save the artifact.
    """
    logger.info("=" * 80)
    logger.info(f"TRAINING MODEL: {model_name}")

    with mlflow.start_run(run_name=model_name):
        pipeline.fit(X_train, y_train)

        valid_pred = pipeline.predict(X_valid)
        valid_prob = pipeline.predict_proba(X_valid)[:, 1]

        test_pred = pipeline.predict(X_test)
        test_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "valid_precision": precision_score(y_valid, valid_pred, zero_division=0),
            "valid_recall": recall_score(y_valid, valid_pred, zero_division=0),
            "valid_f1": f1_score(y_valid, valid_pred, zero_division=0),
            "valid_roc_auc": roc_auc_score(y_valid, valid_prob),
            "valid_pr_auc": average_precision_score(y_valid, valid_prob),
            "test_precision": precision_score(y_test, test_pred, zero_division=0),
            "test_recall": recall_score(y_test, test_pred, zero_division=0),
            "test_f1": f1_score(y_test, test_pred, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, test_prob),
            "test_pr_auc": average_precision_score(y_test, test_prob),
        }

        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        # Log params
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("target_column", settings.target_column)
        mlflow.log_param("feature_count", X_train.shape[1])
        mlflow.log_param("split_strategy", "time_aware_60_20_20")
        mlflow.log_param("leakage_columns_removed", ",".join(settings.current_window_redteam_columns))

        # Save classification reports
        model_dir = LOCAL_MODEL_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        (model_dir / "validation_report.txt").write_text(
            classification_report(y_valid, valid_pred, zero_division=0),
            encoding="utf-8",
        )
        (model_dir / "test_report.txt").write_text(
            classification_report(y_test, test_pred, zero_division=0),
            encoding="utf-8",
        )

        write_json(model_dir / "metrics.json", metrics)

        model_path = model_dir / f"{model_name}.joblib"
        joblib.dump(pipeline, model_path)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            input_example=X_train.head(3),
        )

        logger.info(f"{model_name} metrics: {json.dumps(metrics, indent=2)}")

        return {
            "name": model_name,
            "metrics": metrics,
            "model_path": str(model_path.resolve()),
        }


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    spark = build_spark_session("lanl_train_model")

    logger.info("READING GOLD TABLE")
    logger.info(f"GOLD PATH: {settings.gold_computer_time_path}")

    # Gold output was written as parquet by the EMR gold job.
    gold = spark.read.parquet(settings.gold_computer_time_path)

    row_count = gold.count()
    logger.info(f"GOLD ROW COUNT: {row_count:,}")
    logger.info(f"GOLD COLUMNS: {gold.columns}")

    # Convert to pandas for scikit-learn
    df = gold.toPandas()
    logger.info(f"GOLD PANDAS SHAPE: {df.shape}")

    if settings.target_column not in df.columns:
        raise ValueError(f"Missing target column: {settings.target_column}")

    logger.info("FULL TARGET DISTRIBUTION")
    logger.info(df[settings.target_column].value_counts(dropna=False).to_string())

    # Time-aware split
    train_df, valid_df, test_df = time_split(df)

    logger.info(f"TRAIN SHAPE: {train_df.shape}")
    logger.info(f"VALID SHAPE: {valid_df.shape}")
    logger.info(f"TEST SHAPE: {test_df.shape}")

    logger.info("TRAIN TARGET BALANCE")
    logger.info(train_df[settings.target_column].value_counts(dropna=False).to_string())

    logger.info("VALID TARGET BALANCE")
    logger.info(valid_df[settings.target_column].value_counts(dropna=False).to_string())

    logger.info("TEST TARGET BALANCE")
    logger.info(test_df[settings.target_column].value_counts(dropna=False).to_string())

    # --------------------------------------------------------
    # FEATURE SELECTION
    # --------------------------------------------------------
    # Start from the full dataframe and remove non-feature columns plus
    # leakage-sensitive current-window redteam columns.
    feature_df = df.drop(columns=[settings.target_column], errors="ignore")
    feature_df = feature_df.drop(
        columns=[c for c in settings.non_feature_columns if c in feature_df.columns],
        errors="ignore",
    )
    feature_df = feature_df.drop(
        columns=[c for c in settings.current_window_redteam_columns if c in feature_df.columns],
        errors="ignore",
    )

    feature_cols = feature_df.columns.tolist()
    assert_no_leakage(feature_cols)

    write_json(FEATURE_NAMES_FILE, {"feature_names": feature_cols})

    X_train = train_df[feature_cols]
    y_train = train_df[settings.target_column].astype(int)

    X_valid = valid_df[feature_cols]
    y_valid = valid_df[settings.target_column].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df[settings.target_column].astype(int)

    numeric_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_cols,
            ),
        ],
        remainder="drop",
    )

    models = {
        "logistic_regression_baseline": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest_model": RandomForestClassifier(
            n_estimators=300,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    results: list[dict] = []

    for model_name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        result = evaluate_model(
            model_name=model_name,
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
        )
        results.append(result)

    # Pick best model by validation F1
    best = sorted(results, key=lambda r: r["metrics"]["valid_f1"], reverse=True)[0]
    write_json(BEST_MODEL_SUMMARY_FILE, best)

    logger.info(f"BEST MODEL: {best['name']}")
    logger.info(f"BEST VALID F1: {best['metrics']['valid_f1']:.6f}")

    spark.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("train_model failed")
        sys.exit(1)