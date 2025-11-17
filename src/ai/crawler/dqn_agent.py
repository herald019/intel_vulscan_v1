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
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.lr = lr
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

        # build model only if action_size > 0
        if self.action_size > 0 and self.state_size > 0:
            self.model = self._build_model()
        else:
            self.model = None

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
        # ensure stored as numpy arrays for consistency
        self.memory.append((np.array(state, dtype=np.float32),
                            int(action),
                            float(reward),
                            np.array(next_state, dtype=np.float32),
                            bool(done)))

    # -----------------------------------
    def act(self, state):
        # if no actions available, return 0 safely
        if self.action_size == 0:
            return 0
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        if self.model is None:
            return random.randrange(self.action_size)
        q = self.model.predict(state.reshape(1, -1), verbose=0)
        return int(np.argmax(q[0]))

    # -----------------------------------
    def train_replay(self):
        if self.model is None:
            return
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size), dtype=np.float32)
        targets = np.zeros((self.batch_size, self.action_size), dtype=np.float32)

        for i, (state, action, reward, next_state, done) in enumerate(batch):
            states[i] = state
            q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]
            target = reward
            if not done:
                next_qs = self.model.predict(next_state.reshape(1, -1), verbose=0)[0]
                target += self.gamma * np.max(next_qs)
            q_values[action] = target
            targets[i] = q_values

        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
