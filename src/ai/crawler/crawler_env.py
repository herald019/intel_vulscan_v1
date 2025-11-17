# src/ai/crawler/crawler_env.py
import numpy as np
from zapv2 import ZAPv2
import time

class CrawlerEnv:
    """
    DQN Crawler Environment
    State: [current_idx, visited_flags[]]
    Action: index of next URL
    """

    def __init__(self, target, zap_proxy="http://localhost:8090"):
        self.target = target
        self.zap = ZAPv2(proxies={'http': zap_proxy, 'https': zap_proxy})

        self.urls = []
        self.current_idx = 0
        self.visited = []

    # -----------------------------------------------------
    def reset(self):
        """Spider the site and initialize state"""

        spider_id = self.zap.spider.scan(self.target)
        print("[Crawler] Spidering target...")

        # Wait until spider finishes (sleep to avoid busy loop)
        while True:
            try:
                status = int(self.zap.spider.status(spider_id))
            except Exception:
                status = 100
            if status >= 100:
                break
            time.sleep(0.5)

        # Get discovered URLs
        try:
            results = self.zap.spider.results(spider_id)
        except Exception:
            results = []

        # HANDLE EMPTY RESULTS
        if not results:
            # fallback to target only
            print("[!] Spider returned ZERO URLs! Using fallback:", self.target)
            results = [self.target]

        # Ensure target is included
        if self.target not in results:
            results = [self.target] + results

        # Normalize + remove duplicates while preserving order
        self.urls = list(dict.fromkeys(results))
        self.visited = [0] * len(self.urls)
        self.current_idx = 0

        # Return state + minor info
        state = self._get_state()
        info = {"urls": self.urls}

        return state, info

    # -----------------------------------------------------
    def _get_state(self):
        visited_arr = np.array(self.visited, dtype=np.float32)
        # state vector: [current_idx] + visited_flags
        return np.concatenate(([float(self.current_idx)], visited_arr))

    # -----------------------------------------------------
    def step(self, action_idx):
        """Execute one crawling action"""

        if action_idx < 0 or action_idx >= len(self.urls):
            return self._get_state(), -5.0, True, {"error": "invalid_action"}

        url = self.urls[action_idx]

        try:
            self.zap.urlopen(url)
        except Exception:
            # network/zap error -> penalize slightly but keep running
            pass

        reward = 1.0

        if self.visited[action_idx] == 0:
            reward += 2.0
        else:
            reward -= 1.0

        self.visited[action_idx] = 1
        self.current_idx = action_idx

        # get alerts for this URL; if zap errors, treat as no alerts
        try:
            alerts = self.zap.core.alerts(baseurl=url) or []
        except Exception:
            alerts = []

        reward += len(alerts) * 3.0

        done = False
        if sum(self.visited) >= len(self.visited):
            done = True
        if len(alerts) > 5:
            done = True

        next_state = self._get_state()
        info = {"url": url, "alerts": alerts}

        return next_state, float(reward), done, info

    # -----------------------------------------------------
    def action_space_size(self):
        return len(self.urls)

    def state_space_size(self):
        # state = [current_idx + visited_flags]
        return 1 + max(0, len(self.visited))
    

    # -----------------------------------------------------    
    def get_padded_state(self, target_state_size):
        state = self._get_state()
        if len(state) < target_state_size:
            # pad zeros
            padded = np.zeros(target_state_size, dtype=np.float32)
            padded[:len(state)] = state
            return padded
        else:
            return state[:target_state_size]
