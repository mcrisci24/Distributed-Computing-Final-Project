"""
streamlit_app.py
================

Purpose
-------
A simple but professional UI for manual prediction requests.

Why this matters
----------------
The project rubric requires a publicly accessible web application with
a clear prediction workflow and at least one meaningful visualization.

Design choices
--------------
- Features are grouped into expandable sections by prefix
- Missing values default to 0
- Probability is shown clearly
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from lanl_contracts import FEATURE_NAMES_FILE

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="LANL Threat Predictor", layout="wide")

st.title("LANL Threat Prediction Dashboard")
st.caption("Predict whether a computer will experience red-team activity in the next time window.")

if not FEATURE_NAMES_FILE.exists():
    st.error(f"Feature file missing: {FEATURE_NAMES_FILE}")
    st.stop()

feature_names = json.loads(FEATURE_NAMES_FILE.read_text(encoding="utf-8"))["feature_names"]

# Group features by prefix for cleaner UI
feature_groups = {
    "time": [],
    "auth": [],
    "flows": [],
    "dns": [],
    "proc": [],
    "other": [],
}

for feat in feature_names:
    if feat.startswith("auth_"):
        feature_groups["auth"].append(feat)
    elif feat.startswith("flows_"):
        feature_groups["flows"].append(feat)
    elif feat.startswith("dns_"):
        feature_groups["dns"].append(feat)
    elif feat.startswith("proc_"):
        feature_groups["proc"].append(feat)
    elif feat.startswith("time_"):
        feature_groups["time"].append(feat)
    else:
        feature_groups["other"].append(feat)

inputs = {}

for group_name, feats in feature_groups.items():
    if not feats:
        continue

    with st.expander(f"{group_name.upper()} features", expanded=(group_name in ["auth", "flows"])):
        for feat in feats:
            inputs[feat] = st.number_input(
                label=feat,
                value=0.0,
                step=1.0,
                format="%.4f"
            )

if st.button("Predict"):
    try:
        response = requests.post(API_URL, json=inputs, timeout=30)
        response.raise_for_status()
        result = response.json()

        pred = result["prediction"]
        prob = result["probability_redteam_next_window"]

        st.subheader("Prediction Result")
        st.metric("Predicted Class", pred)
        st.metric("Probability of Red-Team Activity Next Window", f"{prob:.4f}")

        # Simple visualization
        chart_df = pd.DataFrame({
            "class": ["No Red-Team Activity", "Red-Team Activity"],
            "probability": [1 - prob, prob],
        })
        st.bar_chart(chart_df.set_index("class"))

    except Exception as e:
        st.error(f"Prediction request failed: {e}")