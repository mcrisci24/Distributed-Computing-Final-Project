# LANL Threat Prediction Pipeline

A distributed cybersecurity pipeline that ingests multi-source enterprise telemetry from the Los Alamos National Laboratory (LANL) dataset, transforms it through a medallion architecture, engineers time-windowed behavioral features, trains predictive machine learning models, and serves predictions through a live API and web application.

## Project Goal

This project predicts whether a computer will exhibit **red-team activity in the next event window** based on its current behavior across multiple telemetry sources:

- authentication events
- network flows
- DNS lookups
- process lifecycle events

The prediction target is intentionally **future-window compromise risk**, not same-window compromise, to avoid target leakage and better reflect a realistic operational detection setting.

## Prediction Question

**Given a computer’s recent authentication, flow, DNS, and process behavior in the current event-time window, can we predict whether that computer will show red-team activity in the next event-time window?**

## Data Source

This project uses the LANL comprehensive multi-source cybersecurity events dataset, which includes:

- `auth.txt.gz`
- `flows.txt.gz`
- `dns.txt.gz`
- `proc.txt.gz`
- `redteam.txt.gz`

### Important data notes

- Time is stored as **relative event time in seconds**, not real calendar dates.
- One row represents different things depending on the source:
  - auth: one authentication event
  - flows: one network flow
  - dns: one DNS resolution event
  - proc: one process lifecycle event
  - redteam: one labeled compromise event
- Repeated rows are not automatically bad data in cyber telemetry. Some reflect legitimate repeated behavior.

## Architecture

This project is designed around a distributed medallion pipeline:

```text
Bronze (raw S3 files)
    ->
Silver (cleaned typed source tables)
    ->
Gold (computer + time_window feature table)
    ->
Model training + MLflow tracking
    ->
FastAPI prediction service
    ->
Streamlit web application
