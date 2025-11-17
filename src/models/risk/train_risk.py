import os
import joblib
import numpy as np
import pandas as pd
import inspect
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
from src.data_prep import load_dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "risk_models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "risk_model.pkl")
PREPROCESS_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
LABEL_MAP_PATH = os.path.join(MODELS_DIR, "label_map.json")

def prepare_features(df):
    df = df.copy()
    df["scan_duration_seconds"] = df["scan_duration_seconds"].fillna(0.0)
    df["alerts_in_scan"] = df["alerts_in_scan"].fillna(0).astype(float)
    text_col = "alert_name"
    cat_col = "target"
    num_cols = ["scan_duration_seconds", "alerts_in_scan"]
    X = df[[text_col, cat_col] + num_cols]
    risk_labels = sorted(df["risk"].unique().tolist())
    label_map = {lab: i for i, lab in enumerate(risk_labels)}
    y = df["risk"].map(label_map).values
    return X, y, label_map

def build_preprocessor():
    text_pipeline = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=2000))])
    sig = inspect.signature(OneHotEncoder.__init__).parameters
    ohe_kwargs = {}
    if "sparse" in sig:
        ohe_kwargs["sparse"] = False
    elif "sparse_output" in sig:
        ohe_kwargs["sparse_output"] = False
    cat_pipeline = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", **ohe_kwargs))])
    num_pipeline = Pipeline([("scale", StandardScaler())])
    preprocessor = ColumnTransformer(transformers=[
        ("text", text_pipeline, "alert_name"),
        ("cat", cat_pipeline, ["target"]),
        ("num", num_pipeline, ["scan_duration_seconds", "alerts_in_scan"])
    ], remainder="drop", sparse_threshold=0)
    return preprocessor

def train_and_save():
    df = load_dataset()
    if df is None or df.empty:
        print("[!] No data available for training. Run some scans first.")
        return
    X, y, label_map = prepare_features(df)
    preprocessor = build_preprocessor()
    X_trans = preprocessor.fit_transform(X)
    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.2, random_state=42, stratify=stratify)
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=len(label_map), n_estimators=200, learning_rate=0.1, min_data_in_leaf=1, min_data_in_bin=1, verbosity=-1, random_state=42)
    clf.fit(X_train, y_train, feature_name="auto")
    y_pred = clf.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESS_PATH)
    import json
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    print(f"[+] Risk model saved to {MODEL_PATH}")
    print(f"[+] Preprocessor saved to {PREPROCESS_PATH}")
    print(f"[+] Label map saved to {LABEL_MAP_PATH}")
