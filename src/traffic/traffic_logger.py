# src/traffic/traffic_logger.py

import os
import json
from datetime import datetime

# Correct root path
PROJECT_ROOT = os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))
                    )
                )

TRAFFIC_DIR = os.path.join(PROJECT_ROOT, "reports", "traffic")
TRAFFIC_FILE = os.path.join(TRAFFIC_DIR, "traffic.jsonl")

os.makedirs(TRAFFIC_DIR, exist_ok=True)


def log_event(scan_id, event: dict):
    event["scan_id"] = scan_id
    event["timestamp"] = datetime.utcnow().isoformat()

    with open(TRAFFIC_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def log_request(scan_id, url, method, status_code, response_time_ms, response_length, alert_name=None):
    """Detailed HTTP request logging"""

    event = {
        "event": "http_request",
        "url": url,
        "method": method,
        "status_code": status_code,
        "response_time_ms": response_time_ms,
        "response_length": response_length,
        "alert": alert_name,
        "is_alert": alert_name is not None
    }

    log_event(scan_id, event)
