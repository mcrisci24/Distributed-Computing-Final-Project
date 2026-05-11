"""
api_app.py
==========

Purpose
-------
Serve the trained model behind a FastAPI prediction endpoint.

Why this matters
----------------
The project rubric requires a predictive model served through a live endpoint.

How it works
------------
1. Load the best saved model
2. Load the feature list
3. Accept a JSON payload
4. Fill missing features with 0
5. Return prediction + probability
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from lanl_contracts import BEST_MODEL_SUMMARY_FILE, FEATURE_NAMES_FILE

app = FastAPI(title="LANL Threat Prediction API")

# Load feature list
if not FEATURE_NAMES_FILE.exists():
    raise FileNotFoundError(f"Missing feature names file: {FEATURE_NAMES_FILE}")

feature_names = json.loads(FEATURE_NAMES_FILE.read_text(encoding="utf-8"))["feature_names"]

# Load best model metadata
if not BEST_MODEL_SUMMARY_FILE.exists():
    raise FileNotFoundError(f"Missing best model summary file: {BEST_MODEL_SUMMARY_FILE}")

best_model_summary = json.loads(BEST_MODEL_SUMMARY_FILE.read_text(encoding="utf-8"))
model_path = Path(best_model_summary["model_path"])

if not model_path.exists():
    raise FileNotFoundError(f"Missing model file: {model_path}")

model = joblib.load(model_path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: dict):
    """
    Accept a dictionary of feature values.
    Missing features default to 0.

    This is intentionally robust:
    if a user omits some fields, we still create a valid prediction row.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object.")

    # Build one row with every expected feature present
    row = {feature: payload.get(feature, 0) for feature in feature_names}

    X = pd.DataFrame([row], columns=feature_names)

    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0, 1])

    return {
        "prediction": pred,
        "probability_redteam_next_window": prob,
        "model_name": best_model_summary["name"],
    }