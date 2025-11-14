import os
import json
from datetime import datetime

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def iso_now():
    return datetime.now().isoformat()

def write_jsonl(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")
