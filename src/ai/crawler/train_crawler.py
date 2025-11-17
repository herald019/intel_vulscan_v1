# src/ai/crawler/train_crawler.py

import numpy as np
from src.ai.crawler.crawler_env import CrawlerEnv
from src.ai.crawler.dqn_agent import DQNAgent
import os
import time


def train_crawler(target, episodes=20, zap_proxy="http://localhost:8090"):
    # Create env and perform initial spider to collect URLs
    env = CrawlerEnv(target, zap_proxy)
    state, info = env.reset()

    action_size = env.action_space_size()
    state_size = env.state_space_size()

    if action_size == 0:
        print("[!] No URLs found by spider. Cannot train crawler.")
        return

    agent = DQNAgent(state_size, action_size)

    # training loop
    for ep in range(episodes):
        state, info = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < (action_size * 5 + 50):  # safe cap
            action = agent.act(np.array(state, dtype=np.float32))

            next_state, reward, done, _info = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.train_replay()

            total_reward += reward
            state = next_state
            steps += 1

        print(f"[Episode {ep+1}/{episodes}]  Reward: {total_reward:.2f}  ε={agent.epsilon:.4f}")

        # short sleep to avoid very tight loop (helps Zap)
        time.sleep(0.2)

    # Save the model
    model_dir = os.path.join("models", "crawler")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "crawler_dqn.h5")

    if agent.model is not None:
        agent.model.save(model_path)
        print(f"[+] Saved trained crawler model: {model_path}")
    else:
        print("[!] Agent model was not created; nothing saved.")
