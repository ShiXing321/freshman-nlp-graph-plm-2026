import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
logs = ROOT / "运行记录/plm"
logs.mkdir(parents=True, exist_ok=True)
jobs = [("bert_tasks.py", [task, "--epochs", "30", "--lr", ".01" if task == "ptuning" else ".0003"])
        for task in ["pointwise", "dssm", "sbert", "pet", "ptuning"]]
jobs += [("span_task.py", ["--augment", aug, "--epochs", "15"]) for aug in ["none", "negative", "swap"]]
for i, (script, arguments) in enumerate(jobs):
    print(script, arguments, flush=True)
    with (logs / f"bert-{i:02d}.txt").open("w", encoding="utf-8") as f:
        result = subprocess.run([sys.executable, "-u", str(BASE / script), *arguments], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=__import__("os").environ | {"PYTHONUTF8": "1"})
    print("exit", result.returncode, flush=True)
