from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd

from lanl_contracts import (
    SILVER_FILES, GOLD_FILE, WINDOW_SIZE_SECONDS, SOURCE_METADATA
)
from dq_utils import (
    write_json_report,
    missing_value_report,
    full_row_duplicate_report,
    business_key_duplicate_report,
    numeric_distribution_report,
    time_horizon_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DQ_DIR = GOLD_FILE.parent / "dq_reports"
DQ_DIR.mkdir(parents=True, exist_ok=True)


def add_time_window(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df["time_window"] = (df[time_col] // WINDOW_SIZE_SECONDS).astype("Int64")
    return df


def load_silver(name: str) -> pd.DataFrame:
    path = SILVER_FILES[name]
    logger.info(f"Loading silver source: {name} | path={path}")
    if not path.exists():
        raise FileNotFoundError(f"Missing silver file: {path}")
    df = pd.read_csv(path)
    logger.info(f"{name} shape={df.shape}")
    return df


def build_source_overlap_report(source_frames: dict[str, pd.DataFrame]) -> dict:
    report = {}
    for name, df in source_frames.items():
        tw = df["time_window"].dropna()
        report[name] = {
            "min_time_window": int(tw.min()) if len(tw) else None,
            "max_time_window": int(tw.max()) if len(tw) else None,
            "n_time_windows": int(tw.nunique()) if len(tw) else 0,
        }
    return report


def main() -> None:
    auth = add_time_window(load_silver("auth"))
    flows = add_time_window(load_silver("flows"))
    dns = add_time_window(load_silver("dns"))
    proc = add_time_window(load_silver("proc"))
    red = add_time_window(load_silver("redteam"))

    source_frames = {
        "auth": auth,
        "flows": flows,
        "dns": dns,
        "proc": proc,
        "redteam": red,
    }

    # Source-level DQ reports
    for name, df in source_frames.items():
        report = {
            "source_name": name,
            "row_grain": SOURCE_METADATA[name]["row_grain"],
            "time_horizon": time_horizon_report(df, "time"),
            "missing_values": missing_value_report(df),
            "full_row_duplicates": full_row_duplicate_report(df),
            "business_key_duplicates": business_key_duplicate_report(
                df, SOURCE_METADATA[name]["business_key_cols"]
            ),
        }
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            report["numeric_distributions"] = numeric_distribution_report(df, numeric_cols[:10])
        write_json_report(DQ_DIR / f"{name}_dq_report.json", report)

    overlap_report = build_source_overlap_report(source_frames)
    write_json_report(DQ_DIR / "source_time_window_overlap_report.json", overlap_report)
    logger.info(f"Source overlap report: {overlap_report}")

    # AUTH FEATURES
    if "success_flag" not in auth.columns:
        auth["success_flag"] = auth["success"].astype(str).str.upper().eq("SUCCESS").astype(int)
    auth["failure_flag"] = 1 - auth["success_flag"]

    auth_src_features = (
        auth.groupby(["src_computer", "time_window"], dropna=False)
        .agg(
            auth_src_event_count=("time", "size"),
            auth_src_success_count=("success_flag", "sum"),
            auth_src_failure_count=("failure_flag", "sum"),
            auth_src_unique_dst_computers=("dst_computer", "nunique"),
            auth_src_unique_dst_users=("dst_user_domain", "nunique"),
        )
        .reset_index()
        .rename(columns={"src_computer": "computer"})
    )

    auth_dst_features = (
        auth.groupby(["dst_computer", "time_window"], dropna=False)
        .agg(
            auth_dst_event_count=("time", "size"),
            auth_dst_success_count=("success_flag", "sum"),
            auth_dst_failure_count=("failure_flag", "sum"),
            auth_dst_unique_src_computers=("src_computer", "nunique"),
            auth_dst_unique_src_users=("src_user_domain", "nunique"),
        )
        .reset_index()
        .rename(columns={"dst_computer": "computer"})
    )

    # FLOWS FEATURES
    for col in ["duration", "protocol", "packet_count", "byte_count"]:
        flows[col] = pd.to_numeric(flows[col], errors="coerce")

    flows_src_features = (
        flows.groupby(["src_computer", "time_window"], dropna=False)
        .agg(
            flows_out_count=("time", "size"),
            flows_out_unique_dst_computers=("dst_computer", "nunique"),
            flows_out_unique_dst_ports=("dst_port", "nunique"),
            flows_out_total_duration=("duration", "sum"),
            flows_out_total_packets=("packet_count", "sum"),
            flows_out_total_bytes=("byte_count", "sum"),
            flows_out_mean_packets=("packet_count", "mean"),
            flows_out_mean_bytes=("byte_count", "mean"),
        )
        .reset_index()
        .rename(columns={"src_computer": "computer"})
    )

    flows_dst_features = (
        flows.groupby(["dst_computer", "time_window"], dropna=False)
        .agg(
            flows_in_count=("time", "size"),
            flows_in_unique_src_computers=("src_computer", "nunique"),
            flows_in_unique_src_ports=("src_port", "nunique"),
            flows_in_total_duration=("duration", "sum"),
            flows_in_total_packets=("packet_count", "sum"),
            flows_in_total_bytes=("byte_count", "sum"),
            flows_in_mean_packets=("packet_count", "mean"),
            flows_in_mean_bytes=("byte_count", "mean"),
        )
        .reset_index()
        .rename(columns={"dst_computer": "computer"})
    )

    # DNS FEATURES
    dns_features = (
        dns.groupby(["src_computer", "time_window"], dropna=False)
        .agg(
            dns_lookup_count=("time", "size"),
            dns_unique_resolved_computers=("resolved_computer", "nunique"),
        )
        .reset_index()
        .rename(columns={"src_computer": "computer"})
    )

    # PROC FEATURES
    proc["event_type"] = proc["event_type"].astype(str).str.strip().str.upper()
    proc["proc_start_flag"] = proc["event_type"].eq("START").astype(int)
    proc["proc_end_flag"] = proc["event_type"].eq("END").astype(int)

    proc_features = (
        proc.groupby(["computer", "time_window"], dropna=False)
        .agg(
            proc_event_count=("time", "size"),
            proc_start_count=("proc_start_flag", "sum"),
            proc_end_count=("proc_end_flag", "sum"),
            proc_unique_users=("user_domain", "nunique"),
            proc_unique_processes=("process_name", "nunique"),
        )
        .reset_index()
    )
    proc_features["proc_start_end_imbalance"] = (
        proc_features["proc_start_count"] - proc_features["proc_end_count"]
    )

    # REDTEAM CURRENT-WINDOW FLAGS (for ops diagnostics only)
    red_src = (
        red.groupby(["src_computer", "time_window"], dropna=False)
        .agg(redteam_src_event_count=("time", "size"))
        .reset_index()
        .rename(columns={"src_computer": "computer"})
    )

    red_dst = (
        red.groupby(["dst_computer", "time_window"], dropna=False)
        .agg(redteam_dst_event_count=("time", "size"))
        .reset_index()
        .rename(columns={"dst_computer": "computer"})
    )

    # KEYSPACE
    key_frames = [
        auth_src_features[["computer", "time_window"]],
        auth_dst_features[["computer", "time_window"]],
        flows_src_features[["computer", "time_window"]],
        flows_dst_features[["computer", "time_window"]],
        dns_features[["computer", "time_window"]],
        proc_features[["computer", "time_window"]],
        red_src[["computer", "time_window"]],
        red_dst[["computer", "time_window"]],
    ]

    gold = pd.concat(key_frames, ignore_index=True).drop_duplicates().reset_index(drop=True)

    for feature_df in [
        auth_src_features, auth_dst_features, flows_src_features, flows_dst_features,
        dns_features, proc_features, red_src, red_dst
    ]:
        gold = gold.merge(feature_df, on=["computer", "time_window"], how="left")

    for col in gold.columns:
        if col not in ["computer", "time_window"]:
            gold[col] = pd.to_numeric(gold[col], errors="coerce").fillna(0)

    # Derived current-window metrics
    gold["auth_total_events"] = gold["auth_src_event_count"] + gold["auth_dst_event_count"]
    gold["auth_total_failures"] = gold["auth_src_failure_count"] + gold["auth_dst_failure_count"]
    gold["auth_total_successes"] = gold["auth_src_success_count"] + gold["auth_dst_success_count"]

    gold["flows_total_events"] = gold["flows_out_count"] + gold["flows_in_count"]
    gold["flows_total_bytes"] = gold["flows_out_total_bytes"] + gold["flows_in_total_bytes"]
    gold["flows_total_packets"] = gold["flows_out_total_packets"] + gold["flows_in_total_packets"]

    gold["redteam_event_count"] = gold["redteam_src_event_count"] + gold["redteam_dst_event_count"]
    gold["redteam_current_flag"] = (gold["redteam_event_count"] > 0).astype(int)

    gold["auth_failure_ratio"] = gold["auth_total_failures"] / (gold["auth_total_events"] + 1e-9)
    gold["flows_bytes_per_event"] = gold["flows_total_bytes"] / (gold["flows_total_events"] + 1e-9)
    gold["flows_packets_per_event"] = gold["flows_total_packets"] / (gold["flows_total_events"] + 1e-9)

    # FUTURE TARGET
    gold = gold.sort_values(by=["computer", "time_window"]).reset_index(drop=True)
    gold["target_redteam_next_window"] = (
        gold.groupby("computer")["redteam_current_flag"].shift(-1).fillna(0).astype(int)
    )

    # Final DQ
    gold_report = {
        "row_grain": SOURCE_METADATA["gold"]["row_grain"],
        "missing_values": missing_value_report(gold),
        "full_row_duplicates": full_row_duplicate_report(gold),
        "business_key_duplicates": business_key_duplicate_report(
            gold, SOURCE_METADATA["gold"]["business_key_cols"]
        ),
        "time_horizon": {
            "min_time_window": int(gold["time_window"].min()),
            "max_time_window": int(gold["time_window"].max()),
            "n_time_windows": int(gold["time_window"].nunique()),
        },
        "target_distribution": gold["target_redteam_next_window"].value_counts(dropna=False).to_dict(),
    }
    write_json_report(DQ_DIR / "gold_dq_report.json", gold_report)

    GOLD_FILE.parent.mkdir(parents=True, exist_ok=True)
    gold.to_csv(GOLD_FILE, index=False)
    logger.info(f"Saved gold table to {GOLD_FILE}")
    logger.info(f"Final gold shape={gold.shape}")
    logger.info(
        f"Current-window redteam distribution={gold['redteam_current_flag'].value_counts(dropna=False).to_dict()}"
    )
    logger.info(
        f"Next-window target distribution={gold['target_redteam_next_window'].value_counts(dropna=False).to_dict()}"
    )


if __name__ == "__main__":
    main()