# src/ai/crawler/run_crawler.py

import numpy as np
from keras.models import load_model
from src.ai.crawler.crawler_env import CrawlerEnv


def run_crawler(target, zap_proxy="http://localhost:8090"):

    env = CrawlerEnv(target, zap_proxy)
    model_path = "models/crawler/crawler_dqn.h5"

    if not os.path.exists(model_path):
        print("[!] No trained crawler model found. Train first using --train-crawler")
        return

    model = load_model(model_path)

    state = env.reset()
    done = False

    print("\n[*] Running Smart Crawler...\n")

    steps = 0
    while not done and steps < 50:
        qs = model.predict(state.reshape(1, -1), verbose=0)
        action = np.argmax(qs)

        next_state, reward, done, info = env.step(action)
        print(f"[Step {steps}] → {info['url']} | reward={reward}")

        state = next_state
        steps += 1

    print("\n[+] Crawler finished.")
