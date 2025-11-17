# src/traffic/parse_traffic.py

import os
import json
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAFFIC_DIR = os.path.join(PROJECT_ROOT, "reports", "traffic")


def load_jsonl(file_path):
    if not os.path.exists(file_path):
        return []

    events = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except:
                continue

    return events


def safe_series(df, key, default=0):
    """Ensures df[key] always returns a Series."""
    if key not in df:
        return pd.Series([default] * len(df))
    return df[key].fillna(default)


def build_sequences_from_jsonl(seq_len=20):
    """Build LSTM sequences from traffic logs stored in /reports/traffic/*.jsonl"""

    if not os.path.exists(TRAFFIC_DIR):
        print("[!] No traffic directory found.")
        return None

    all_events = []

    for fname in os.listdir(TRAFFIC_DIR):
        if fname.endswith(".jsonl"):
            events = load_jsonl(os.path.join(TRAFFIC_DIR, fname))
            all_events.extend(events)

    if len(all_events) == 0:
        print("[!] No traffic events found.")
        return None

    df = pd.DataFrame(all_events)

    # Normalize column names
    if "response_time_ms" in df.columns:
        df["response_time"] = df["response_time_ms"]
    if "response_length" in df.columns:
        df["payload_size"] = df["response_length"]

    # Ensure numeric columns exist
    df["status_code"] = safe_series(df, "status_code", 0).astype(float)
    df["response_time"] = safe_series(df, "response_time", 0).astype(float)
    df["payload_size"] = safe_series(df, "payload_size", 0).astype(float)

    # boolean flags
    df["is_alert"] = safe_series(df, "is_alert", 0).astype(int)

    # Sort by timestamp if present
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    # Select features used by LSTM
    features = ["status_code", "response_time", "payload_size", "is_alert"]
    data = df[features].to_numpy()

    # Build sequences
    X = []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])

    return np.array(X, dtype=np.float32)
