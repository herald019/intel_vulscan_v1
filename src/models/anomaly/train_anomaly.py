# src/models/anomaly/train_anomaly.py
import os
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from src.traffic.parse_traffic import build_sequences_from_jsonl

PROJECT_ROOT =os.path.dirname( 
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))
                        )
                    )
                )
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "anomaly_models")
MODEL_PATH = os.path.join(PROJECT_ROOT, MODELS_DIR, "anomaly_model_ver1.h5")


def build_model(seq_len):
    model = Sequential()

    # Encoder
    model.add(LSTM(64, activation="relu", input_shape=(seq_len, 4)))
    model.add(RepeatVector(seq_len))  # repeat for decoder input

    # Decoder
    model.add(LSTM(64, activation="relu", return_sequences=True))
    model.add(TimeDistributed(Dense(4)))  # output full sequence

    model.compile(optimizer="adam", loss="mse")
    return model


def train_and_save(seq_len=20, batch_size=16, epochs=10):
    print("[*] Training anomaly detection model...")

    X = build_sequences_from_jsonl(seq_len=seq_len)
    if X is None or len(X) == 0:
        print("[!] Not enough traffic data. Run scans first.")
        return

    X = np.array(X, dtype=np.float32)

    if len(X) < 5:
        print(f"[!] Not enough data ({len(X)} samples). Need at least 5.")
        return

    # Normalize
    X_norm = (X - X.min()) / (X.max() - X.min() + 1e-6)

    model = build_model(seq_len)

    if len(X_norm) < 40:
        print("[*] Small dataset — training WITHOUT validation split.")
        model.fit(X_norm, X_norm, epochs=epochs, batch_size=batch_size, verbose=1)
    else:
        model.fit(
            X_norm, X_norm,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)

    print(f"[+] Anomaly detection model saved to {MODEL_PATH}")
