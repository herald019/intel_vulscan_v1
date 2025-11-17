import os
import joblib
import json
import numpy as np
from src.data_prep import load_dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "risk_models")
MODEL_PATH = os.path.join(MODELS_DIR, "risk_model.pkl")
PREPROCESS_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
LABEL_MAP_PATH = os.path.join(MODELS_DIR, "label_map.json")

def demo_predict_latest():
    # load dataset (we will pick latest scan alerts)
    df = load_dataset()
    if df is None or df.empty:
        print("[!] No data to predict on.")
        return
    # pick latest scan id
    latest_scan = df.iloc[0].scan_id
    subset = df[df.scan_id == latest_scan]
    if subset.empty:
        print("[!] No alerts in latest scan.")
        return
    # load artifacts
    clf = joblib.load(MODEL_PATH)
    pre = joblib.load(PREPROCESS_PATH)
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    inv_map = {v: k for k, v in label_map.items()}
    X = subset[["alert_name", "target", "scan_duration_seconds", "alerts_in_scan"]]
    X_trans = pre.transform(X)
    preds = clf.predict(X_trans)
    probs = clf.predict_proba(X_trans) if hasattr(clf, "predict_proba") else None
    for i, row in subset.iterrows():
        label = inv_map.get(int(preds[list(subset.index).index(i)]), "Unknown")
        print(f"Alert: {row.alert_name} -> Predicted Risk: {label}")
