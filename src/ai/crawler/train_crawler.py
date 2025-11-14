# src/ai/crawler/train_crawler.py

import numpy as np
from src.ai.crawler.crawler_env import CrawlerEnv
from src.ai.crawler.dqn_agent import DQNAgent
import os


def train_crawler(target, episodes=20, zap_proxy="http://localhost:8090"):
    env = CrawlerEnv(target, zap_proxy)

    state_size = env.state_space_size()
    action_size = env.action_space_size()

    agent = DQNAgent(state_size, action_size)

    for ep in range(episodes):
        state = env.reset()

        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)

            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.train_replay()

            total_reward += reward
            state = next_state

        print(f"[Episode {ep+1}/{episodes}]  Reward: {total_reward:.2f}  ε={agent.epsilon:.4f}")

    # Save the model
    model_dir = os.path.join("models", "crawler")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "crawler_dqn.h5")

    agent.model.save(model_path)
    print(f"[+] Saved trained crawler model: {model_path}")
