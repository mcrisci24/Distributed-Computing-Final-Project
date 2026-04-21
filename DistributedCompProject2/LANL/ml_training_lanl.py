"""
ml_training_lanl.py
===================

Purpose
-------
Train benchmark and stronger ML models using the GOLD feature table.

Why this matters
----------------
This is the predictive modeling layer of the project.

Key maturity choices
--------------------
1. We predict FUTURE redteam activity, not same-window activity.
2. We explicitly remove leakage columns.
3. We split by TIME WINDOW, not random shuffle, so evaluation is more realistic.
4. We report class balance and metrics appropriate for imbalanced classification.

Target
------
target_redteam_next_window
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
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

from lanl_contracts import (
    GOLD_OUTPUT,
    MODEL_DIR,
    FEATURE_NAMES_FILE,
    BEST_MODEL_SUMMARY_FILE,
    TARGET_COLUMN,
    NON_FEATURE_COLUMNS,
    REDTEAM_CURRENT_COLUMNS,
)
from spark_runtime import build_spark_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MLFLOW_DIR = MODEL_DIR.parent / "mlruns"
MLFLOW_TRACKING_URI = MLFLOW_DIR.as_uri()
MLFLOW_EXPERIMENT_NAME = "lanl_threat_prediction"

RANDOM_STATE = 42


def assert_no_leakage(feature_cols: list[str]) -> None:
    """
    Prevent current-window redteam columns from entering the model.
    """
    leaking = [c for c in feature_cols if c in REDTEAM_CURRENT_COLUMNS]
    if leaking:
        raise ValueError(f"Leakage columns found in feature set: {leaking}")


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-aware split:
    - earliest 60% of windows = train
    - next 20% = validation
    - latest 20% = test

    This is more realistic than random row shuffling for event-time problems.
    """
    windows = sorted(df["time_window"].dropna().unique().tolist())

    if len(windows) < 5:
        raise ValueError("Not enough time windows for a meaningful time-aware split.")

    train_cut_idx = int(len(windows) * 0.60)
    valid_cut_idx = int(len(windows) * 0.80)

    train_cut = windows[train_cut_idx]
    valid_cut = windows[valid_cut_idx]

    train_df = df[df["time_window"] <= train_cut].copy()
    valid_df = df[(df["time_window"] > train_cut) & (df["time_window"] <= valid_cut)].copy()
    test_df = df[df["time_window"] > valid_cut].copy()

    return train_df, valid_df, test_df


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    spark = build_spark_session("lanl_ml_training")

    logger.info("Loading GOLD Delta for model training")
    gold = spark.read.format("delta").load(str(GOLD_OUTPUT))

    logger.info(f"GOLD row count: {gold.count()}")
    logger.info(f"GOLD columns: {gold.columns}")

    # Convert to pandas for scikit-learn modeling.
    # This is acceptable because the heavy transformation happened in Spark.
    df = gold.toPandas()

    logger.info(f"Converted GOLD to pandas shape: {df.shape}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    # Keep a clean feature set
    feature_df = df.drop(columns=[TARGET_COLUMN], errors="ignore")
    feature_df = feature_df.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in feature_df.columns], errors="ignore")
    feature_df = feature_df.drop(columns=[c for c in REDTEAM_CURRENT_COLUMNS if c in feature_df.columns], errors="ignore")

    feature_cols = feature_df.columns.tolist()
    assert_no_leakage(feature_cols)

    # Save feature names for serving later
    write_json(FEATURE_NAMES_FILE, {"feature_names": feature_cols})

    # Create time-aware splits
    train_df, valid_df, test_df = time_split(df)

    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Validation shape: {valid_df.shape}")
    logger.info(f"Test shape: {test_df.shape}")

    logger.info(f"Train target distribution:\n{train_df[TARGET_COLUMN].value_counts(dropna=False)}")
    logger.info(f"Validation target distribution:\n{valid_df[TARGET_COLUMN].value_counts(dropna=False)}")
    logger.info(f"Test target distribution:\n{test_df[TARGET_COLUMN].value_counts(dropna=False)}")

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COLUMN].astype(int)

    X_valid = valid_df[feature_cols]
    y_valid = valid_df[TARGET_COLUMN].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COLUMN].astype(int)

    numeric_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_cols),
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

    results = []

    for model_name, model in models.items():
        logger.info(f"Training model: {model_name}")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

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

            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("target_column", TARGET_COLUMN)
            mlflow.log_param("feature_count", len(feature_cols))
            mlflow.log_param("split_strategy", "time_aware_60_20_20")
            mlflow.log_param("leakage_columns_removed", ",".join(REDTEAM_CURRENT_COLUMNS))

            report_dir = MODEL_DIR / model_name
            report_dir.mkdir(parents=True, exist_ok=True)

            (report_dir / "validation_report.txt").write_text(
                classification_report(y_valid, valid_pred, zero_division=0),
                encoding="utf-8",
            )
            (report_dir / "test_report.txt").write_text(
                classification_report(y_test, test_pred, zero_division=0),
                encoding="utf-8",
            )

            write_json(report_dir / "metrics.json", metrics)

            model_path = report_dir / f"{model_name}.joblib"
            joblib.dump(pipeline, model_path)

            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            logger.info(f"{model_name} metrics: {metrics}")

            results.append({
                "name": model_name,
                "metrics": metrics,
                "model_path": str(model_path),
            })

    best = sorted(results, key=lambda r: r["metrics"]["valid_f1"], reverse=True)[0]
    write_json(BEST_MODEL_SUMMARY_FILE, best)

    logger.info(f"Best model: {best['name']}")
    logger.info(f"Best model validation F1: {best['metrics']['valid_f1']}")

    spark.stop()


if __name__ == "__main__":
    main()