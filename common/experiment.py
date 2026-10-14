"""Shared bookkeeping; learning models stay in their own task directories."""
import json
import random
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def seed_all(seed=42):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Run:
    def __init__(self, folder, name, config):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / f"{name}.json"
        self.started = time.perf_counter()
        self.data = {"name": name, "config": config, "started": datetime.now().astimezone().isoformat(),
                     "status": "running", "epochs": [], "device": config.get("device_used") or str(device())}
        self.save()

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def epoch(self, **metrics):
        self.data["epochs"].append(metrics)
        self.save()
        print(json.dumps(metrics, ensure_ascii=False), flush=True)

    def finish(self, **metrics):
        self.data.update(status="completed", elapsed_seconds=time.perf_counter() - self.started, **metrics)
        self.save()
        print(self.path, flush=True)


def download(url, dest, timeout=300):
    """curl works with the Windows certificate store and retries failed transfers."""
    import subprocess
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    subprocess.run(["curl.exe", "--fail", "-L", "--retry", "2", "--connect-timeout", "20",
                    "--max-time", str(timeout), url, "-o", str(tmp)], check=True)
    tmp.replace(dest)
    return dest
