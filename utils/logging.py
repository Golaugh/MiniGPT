import os
import json
from datetime import datetime

class StatsLogger:
    def __init__(self, log_dir="log", log_filename="stats"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{log_filename}_{timestamp}.jsonl")

    def log(self, stats: dict):
        with open(self.log_path, "a", encoding="utf-8") as f:
            json.dump(stats, f)
            f.write("\n")

    def log_console(self, stats: dict):
        formatted = " | ".join(
            f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
            for key, value in stats.items()
        )
        print(f"[LOG] {formatted}")