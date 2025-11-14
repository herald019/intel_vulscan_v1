# src/ai/crawler/dqn_agent.py

import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        lr=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=5000,
        batch_size=32
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

        self.model = self._build_model()

    # -----------------------------------
    def _build_model(self):
        model = Sequential()
        model.add(Dense(256, activation="relu", input_dim=self.state_size))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(optimizer=Adam(learning_rate=self.lr), loss="mse")
        return model

    # -----------------------------------
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # -----------------------------------
    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        q = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q[0])

    # -----------------------------------
    def train_replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        states = []
        targets = []

        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                next_qs = self.model.predict(next_state.reshape(1, -1), verbose=0)[0]
                target += self.gamma * np.max(next_qs)

            q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]
            q_values[action] = target

            states.append(state)
            targets.append(q_values)

        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
