"""Reproduce the saved baseline PLM experiments before the expanded comparisons."""
import json
import os
import subprocess
import sys
from pathlib import Path
from run_resume import completed_artifacts
ROOT = Path(__file__).resolve().parents[1]
P = "transformers_tasks/local_runs/"


def jobs():
    for task in ["classification", "pointwise", "dssm", "sbert", "simcse", "pet", "ptuning"]:
        command = [P + "bert_tasks.py", task, "--epochs", "20" if task == "classification" else "5" if task == "simcse" else "30" if task in ["pet", "ptuning"] else "15"]
        if task == "simcse":
            command += ["--limit", "10000"]
        if task == "ptuning":
            command += ["--lr", ".01"]
        yield command
    for arm in ["none", "negative", "swap"]:
        yield [P + "span_task.py", "--augment", arm]
    yield [P + "reward_task.py", "--limit", "1000", "--epochs", "5"]
    for task in ["qa", "filling"]:
        yield [P + "seq2seq_tasks.py", task, "--limit", "1000", "--epochs", "10", "--batch", "8", "--lr", ".0005"]
    yield [P + "llm_tasks.py", "zero-shot"]
    yield [P + "llm_tasks.py", "mbti"]
    yield [P + "llm_tasks.py", "sft", "--limit", "400", "--epochs", "2"]
    yield [P + "llm_tasks.py", "cpt", "--limit", "400", "--epochs", "1"]
    yield [P + "llm_tasks.py", "scratch", "--limit", "1000", "--epochs", "5", "--batch", "8", "--accumulate", "1", "--lr", ".0005"]
    yield [P + "llm_tasks.py", "scratch-sft", "--limit", "400", "--epochs", "3", "--batch", "8", "--accumulate", "1"]
    yield [P + "ppo_task.py"]


if __name__ == "__main__":
    folder = ROOT / "运行记录/plm-reproduce"
    folder.mkdir(parents=True, exist_ok=True)
    index = folder / "history.jsonl"
    history = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()] if index.exists() else []
    done = {json.dumps(row["command"]) for row in history if row["exit_code"] == 0}
    for i, command in enumerate(jobs()):
        if json.dumps(command) in done and completed_artifacts(command):
            continue
        print(i, command, flush=True)
        with (folder / f"{i:03d}.txt").open("w", encoding="utf-8") as log:
            proc = subprocess.run([sys.executable, "-X", "utf8", "-u", *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=os.environ | {"PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1"})
        with index.open("a", encoding="utf-8") as out:
            out.write(json.dumps({"command": command, "exit_code": proc.returncode}, ensure_ascii=False) + "\n")
        if proc.returncode:
            raise SystemExit(proc.returncode)
