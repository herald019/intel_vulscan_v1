# src/ai/crawler/crawler_env.py

import numpy as np
import random
from zapv2 import ZAPv2


class CrawlerEnv:
    """
    Environment for the DQN Smart Crawler.
    State = flattened representation of (current_index, visited_flags)
    Action = index of next URL to visit
    Reward = based on alerts, new pages discovered, response characteristics
    """

    def __init__(self, target, zap_proxy="http://localhost:8090"):
        self.target = target
        self.zap = ZAPv2(proxies={'http': zap_proxy, 'https': zap_proxy})

        # State vars
        self.urls = []
        self.current_idx = 0
        self.visited = []

        # After reset, spider gathers initial list of URLs
        self.reset()

    # -----------------------------------
    def reset(self):
        """
        Spider the site FIRST to get all URLs.
        """
        spider_id = self.zap.spider.scan(self.target)
        while int(self.zap.spider.status(spider_id)) < 100:
            pass

        # Collect discovered URLs
        results = self.zap.spider.results(spider_id)
        if self.target not in results:
            results = [self.target] + results

        self.urls = list(dict.fromkeys(results))   # remove duplicates
        self.visited = [0] * len(self.urls)
        self.current_idx = 0

        return self._get_state()

    # -----------------------------------
    def _get_state(self):
        """
        State is represented as:
        [current_index, visited_0, visited_1, ... visited_n]
        """
        visited_arr = np.array(self.visited, dtype=np.float32)
        return np.concatenate(([self.current_idx], visited_arr))

    # -----------------------------------
    def step(self, action_idx):
        """
        Execute the action: visit URL at index action_idx
        """
        if action_idx < 0 or action_idx >= len(self.urls):
            # illegal action
            return self._get_state(), -5, True, {"error": "invalid_action"}

        url = self.urls[action_idx]

        # CALL ZAP
        try:
            self.zap.urlopen(url)
        except:
            pass

        # REWARD: base reward for exploring
        reward = 1.0

        # REWARD: new visit?
        if self.visited[action_idx] == 0:
            reward += 2.0            # exploring new URL is good
        else:
            reward -= 1.0            # revisiting gives penalty

        self.visited[action_idx] = 1
        self.current_idx = action_idx

        # Get any alerts for this URL
        alerts = self.zap.core.alerts(baseurl=url)
        reward += len(alerts) * 3.0   # alerts are valuable

        # Episode termination triggers
        done = False
        if sum(self.visited) == len(self.visited):
            done = True
        if len(alerts) > 5:           # too many = unstable
            done = True

        next_state = self._get_state()
        return next_state, reward, done, {"url": url, "alerts": alerts}

    # -----------------------------------
    def action_space_size(self):
        return len(self.urls)

    def state_space_size(self):
        # state = [current_idx + visited_flags]
        return 1 + len(self.urls)
