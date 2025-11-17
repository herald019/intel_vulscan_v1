# src/ai/crawler/run_crawler.py

import os
import time
import numpy as np
from keras.models import load_model

from src.ai.crawler.crawler_env import CrawlerEnv

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "crawler")
MODEL_PATH = os.path.join(MODELS_DIR, "crawler_dqn.h5")


def run_crawler(target):
    if not os.path.exists(MODEL_PATH):
        print("[!] No crawler model found. Train using --train-crawler")
        return

    print(f"[+] Loading trained crawler model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    env = CrawlerEnv(target)
    state, info = env.reset()

    print("[+] URLs discovered:", len(info.get("urls", [])))
    print("[+] Running Smart Crawler...")

    total_reward = 0.0

    for step in range(100):
        state_input = np.array([state], dtype=np.float32)

        q_values = model.predict(state_input, verbose=0)
        action = int(np.argmax(q_values[0]))

        next_state, reward, done, info = env.step(action)

        print(f"[Step {step}] Visit={info.get('url')} | Reward={reward}")

        total_reward += reward
        state = next_state

        if done:
            print("[+] Episode finished.")
            break

        time.sleep(0.25)

    print("\n[+] Crawler completed.")
    print("[+] Total reward:", total_reward)
