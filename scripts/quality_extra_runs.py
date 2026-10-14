"""Additional parameter studies and pretrained UIE runs after the main queue."""
import ctypes
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from run_resume import completed_artifacts
ROOT = Path(__file__).resolve().parents[1]


def jobs():
    for task, ds in [("task1", "Cora"), ("task2", "Cora"), ("task3", "MUTAG"), ("task3", "ZINC")]:
        for layers, lr in [(1, .01 if task != "task3" else .001), (3, .01 if task != "task3" else .001),
                           (2, .001 if task != "task3" else .0001), (2, .1 if task != "task3" else .01)]:
            model = "gin" if task == "task3" else "gcn"
            command = [f"graph_beginner/{task}/code/train.py", "--dataset", ds, "--model", model,
                       "--layers", str(layers), "--lr", str(lr), "--epochs", "100" if task == "task1" else "50" if task == "task2" else "30",
                       "--tag", "_ablation"]
            if task == "task3":
                command += ["--pool", "mean", "--mode", "batch", "--limit", "2000" if ds == "ZINC" else "0"]
            yield command
    for arm in ["none", "negative", "swap", "fill"]:
        yield ["transformers_tasks/local_runs/span_task.py", "--augment", arm, "--pretrained-uie",
               "--epochs", "8", "--batch", "8", "--lr", "0.00002"]
    for task in ["qa", "filling"]:
        yield ["transformers_tasks/local_runs/seq2seq_tasks.py", task, "--limit", "5000", "--epochs", "8",
               "--batch", "8", "--lr", "0.0003", "--eval-limit", "500", "--tag", "_expanded"]
    yield ["transformers_tasks/local_runs/llm_tasks.py", "zero-shot", "--eval-size", "100", "--tag", "_expanded"]
    yield ["transformers_tasks/local_runs/ppo_task.py", "--steps", "48", "--heldout-eval", "--tag", "_expanded"]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        handle = ctypes.windll.kernel32.OpenProcess(0x00100000, False, int(sys.argv[1]))
        if handle:
            ctypes.windll.kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
            ctypes.windll.kernel32.CloseHandle(handle)
    folder = ROOT / "运行记录/quality-extra"
    folder.mkdir(parents=True, exist_ok=True)
    index = folder / "history.jsonl"
    history = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()] if index.exists() else []
    done = {json.dumps(row["command"]) for row in history if row["exit_code"] == 0}
    for i, command in enumerate(jobs()):
        if json.dumps(command) in done and completed_artifacts(command):
            continue
        print(i, command, flush=True)
        with (folder / f"{i:03d}.txt").open("w", encoding="utf-8") as log:
            proc = subprocess.run([sys.executable, "-X", "utf8", "-u", *command], cwd=ROOT,
                                  stdout=log, stderr=subprocess.STDOUT, env=os.environ | {"PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1"})
        with index.open("a", encoding="utf-8") as out:
            out.write(json.dumps({"command": command, "exit_code": proc.returncode, "finished": datetime.now().astimezone().isoformat()}, ensure_ascii=False) + "\n")
        print(i, "exit", proc.returncode, flush=True)
