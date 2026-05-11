"""
test_project.py
===============

Purpose
-------
Minimal end-to-end test for the live prediction API.

What it does
------------
1. Sends one sample request to the /predict endpoint
2. Prints the response
3. Exits 0 on success, non-zero on failure
"""

from __future__ import annotations

import sys
import requests

URL = "http://127.0.0.1:8000/predict"

# A minimal payload.
# Missing features should default to 0 inside the API.
payload = {
    "auth_total_events": 10,
    "auth_total_failures": 2,
    "flows_total_events": 30,
    "flows_total_bytes": 5000,
    "dns_lookup_count": 4,
    "proc_event_count": 7,
}

try:
    resp = requests.post(URL, json=payload, timeout=30)

    if resp.status_code != 200:
        print(f"FAIL: Got status {resp.status_code}")
        print(resp.text)
        sys.exit(1)

    result = resp.json()
    print("PASS")
    print(result)
    sys.exit(0)

except Exception as e:
    print(f"FAIL: Request failed with error: {e}")
    sys.exit(1)