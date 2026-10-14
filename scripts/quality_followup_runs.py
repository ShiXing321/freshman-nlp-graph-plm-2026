"""Recheck ranking metrics and run targeted learning-rate/casing studies."""
import argparse
import ctypes
import itertools
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from run_resume import completed_artifacts
ROOT = Path(__file__).resolve().parents[1]


def jobs():
    yield ["graph_beginner/task4/code/train.py", "--model", "conve", "--epochs", "20", "--limit", "0",
        "--eval-limit", "200", "--test-limit", "0", "--batch", "1024", "--negatives", "8", "--tag", "_full"]
    for kind in ["cnn", "rnn"]:
        yield ["nlp-beginner/legacy/train_torch.py", "classification", "--data-name", "sentiment_kaggle",
            "--model", kind, "--epochs", "8", "--batch", "64", "--glove", "nlp-beginner/legacy/data/raw/glove/glove.6B.50d.txt"]
    yield ["transformers_tasks/local_runs/span_task.py", "--augment", "fill", "--pretrained-uie",
        "--epochs", "8", "--batch", "8", "--lr", "0.00002", "--filling-checkpoint", "t5_filling_expanded"]
    for ds, kind, mode in itertools.product(["Cora", "CiteSeer", "Flickr"], ["gcn", "gat", "sage", "gin"], ["full", "sampled"]):
        yield ["graph_beginner/task2/code/train.py", "--dataset", ds, "--model", kind,
            "--mode", mode, "--epochs", "50", "--hidden", "16" if ds == "Flickr" else "32", "--tag", "_isolated"]
    for kind, mode in itertools.product(["gcn", "gat", "sage", "gin"], ["full", "sampled"]):
        yield ["graph_beginner/task2/code/train.py", "--dataset", "Flickr", "--model", kind,
            "--mode", mode, "--epochs", "50", "--hidden", "16", "--lr", ".001", "--tag", "_tuned"]
    for layers, lr in [(1, .01), (3, .01), (2, .001), (2, .1)]:
        yield ["graph_beginner/task2/code/train.py", "--dataset", "Cora", "--model", "gcn",
            "--layers", str(layers), "--lr", str(lr), "--epochs", "50", "--tag", "_ablation"]
    yield ["nlp-beginner/legacy/train_torch.py", "ner", "--model", "bilstm-crf", "--glove",
        "nlp-beginner/legacy/data/raw/glove/glove.6B.50d.txt", "--epochs", "30", "--hidden", "128",
        "--max-length", "512", "--batch", "32", "--use-case", "--tag", "_casing"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, default=0)
    args = parser.parse_args()
    if args.wait_pid:
        handle = ctypes.windll.kernel32.OpenProcess(0x00100000, False, args.wait_pid)
        if handle:
            ctypes.windll.kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
            ctypes.windll.kernel32.CloseHandle(handle)
    folder = ROOT / "运行记录/quality-followup"
    folder.mkdir(parents=True, exist_ok=True)
    index = folder / "history.jsonl"
    old = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()] if index.exists() else []
    done = {json.dumps(row["command"]) for row in old if row["exit_code"] == 0}
    failures = []
    for i, command in enumerate(jobs()):
        if json.dumps(command) in done and completed_artifacts(command):
            continue
        if "--filling-checkpoint" in command:
            for source, target in [("运行记录/quality/mask_fill_samples.json", "运行记录/诊断/mask_fill_baseline_samples.json"),
                ("transformers_tasks/local_runs/results/uie_base_fill_v1.json", "运行记录/诊断/uie_fill_baseline_result.json")]:
                path = ROOT / source
                if path.exists() and not (ROOT / target).exists():
                    shutil.copyfile(path, ROOT / target)
        print(datetime.now().astimezone().isoformat(), i, command, flush=True)
        with (folder / f"{i:03d}.txt").open("w", encoding="utf-8") as log:
            p = subprocess.run([sys.executable, "-X", "utf8", "-u", *command], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, env=os.environ | {"PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1"})
        row = {"command": command, "exit_code": p.returncode, "finished": datetime.now().astimezone().isoformat()}
        with index.open("a", encoding="utf-8") as out:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(i, "exit", p.returncode, flush=True)
        if p.returncode:
            failures.append(command)
    (folder / "state.json").write_text(json.dumps({"status": "completed" if not failures else "failed", "failures": failures}, indent=2), encoding="utf-8")
    raise SystemExit(bool(failures))
