import os
import numpy as np
import joblib
from src.traffic.parse_traffic import build_sequences_from_jsonl
from keras.models import load_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "anomaly")
MODEL_PATH = os.path.join(MODELS_DIR, "anomaly_model.h5")
MEAN_PATH = os.path.join(MODELS_DIR, "anomaly_mean.npy")
STD_PATH = os.path.join(MODELS_DIR, "anomaly_std.npy")

def load_norm():
    mean = np.load(MEAN_PATH)
    std = np.load(STD_PATH)
    return mean, std

def run_detection(seq_len=50, threshold=None):
    if not os.path.exists(MODEL_PATH):
        print("[!] No trained anomaly model found. Train first with --train-anomaly")
        return
    X = build_sequences_from_jsonl(seq_len=seq_len)
    if X is None or len(X) == 0:
        print("[!] No traffic sequences found.")
        return
    mean, std = load_norm()
    X_norm = (X - mean) / std
    model = load_model(MODEL_PATH)
    recon = model.predict(X_norm)
    mse = np.mean(np.square(recon - X_norm), axis=(1,2))  # per-sequence error
    if threshold is None:
        # heuristic threshold: mean + 3*std of train errors (we use current data)
        threshold = mse.mean() + 3 * mse.std()
    print(f"Using threshold = {threshold:.6f}")
    for i, err in enumerate(mse):
        print(f"Sequence {i}: MSE={err:.6f} -> {'ANOMALY' if err>threshold else 'normal'}")
    return mse, threshold
