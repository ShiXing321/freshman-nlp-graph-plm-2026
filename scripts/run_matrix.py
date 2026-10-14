"""Sequential experiments with durable output and resumable completed runs."""
import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def matrix(group):
    nlp = "nlp-beginner/legacy/"
    graph = "graph_beginner/"
    jobs = []
    if group == "nlp":
        for features, ngram, loss, lr, batch, shuffle in [
            ("count", 1, "softmax", .1, 64, True), ("binary", 1, "softmax", .1, 64, True),
            ("tfidf", 1, "softmax", .5, 64, True), ("tfidf", 2, "softmax", .5, 64, True),
            ("tfidf", 1, "logistic", .5, 64, True), ("tfidf", 1, "softmax", .05, 64, True),
            ("tfidf", 1, "softmax", .5, 0, True), ("tfidf", 1, "softmax", .05, 1, True),
            ("tfidf", 1, "softmax", .5, 64, False)]:
            jobs.append([nlp + "train_numpy.py", "--features", features, "--ngram", str(ngram), "--loss", loss, "--lr", str(lr), "--batch", str(batch), "--epochs", "20"] + ([] if shuffle else ["--no-shuffle"]))
        for model in ["softmax", "cnn", "rnn"]:
            jobs.append([nlp + "train_torch.py", "classification", "--model", model, "--epochs", "8"])
        jobs.append([nlp + "train_torch.py", "matching", "--model", "esim", "--limit", "20000", "--epochs", "5"])
    elif group in ["graph-small", "graph-flickr"]:
        datasets = ["Cora", "CiteSeer"] if group == "graph-small" else ["Flickr"]
        for task, epochs in [("task1", "100"), ("task2", "50")]:
            for ds, model, mode in itertools.product(datasets, ["gcn", "gat", "sage", "gin"], ["full", "sampled"]):
                jobs.append([graph + task + "/code/train.py", "--dataset", ds, "--model", model, "--mode", mode, "--epochs", epochs, "--hidden", "16" if ds == "Flickr" else "32"])
        if group == "graph-small":
            for layers, lr in [(1, .01), (3, .01), (2, .001), (2, .1)]:
                jobs.append([graph + "task1/code/train.py", "--dataset", "Cora", "--model", "gcn", "--layers", str(layers), "--lr", str(lr), "--epochs", "100"])
    elif group in ["graph-tu", "graph-zinc"]:
        ds = "MUTAG" if group == "graph-tu" else "ZINC"
        for model, pool, mode in itertools.product(["gcn", "gat", "sage", "gin"], ["mean", "max", "min"], ["full", "batch"]):
            jobs.append([graph + "task3/code/train.py", "--dataset", ds, "--model", model, "--pool", pool, "--mode", mode, "--epochs", "30", "--limit", "0" if ds == "MUTAG" else "2000"])
    elif group == "kge":
        for model in ["transe", "rotate", "conve"]:
            jobs.append([graph + "task4/code/train.py", "--model", model, "--epochs", "5", "--limit", "20000", "--eval-limit", "50"])
    return jobs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("group", choices=["nlp", "graph-small", "graph-flickr", "graph-tu", "graph-zinc", "kge"])
    p.add_argument("--retry", action="store_true")
    args = p.parse_args()
    folder = ROOT / "运行记录" / args.group
    folder.mkdir(parents=True, exist_ok=True)
    index = folder / "runs.json"
    history = json.loads(index.read_text(encoding="utf-8")) if index.exists() else []
    completed = {json.dumps(h["command"]) for h in history if h["exit_code"] == 0}
    for i, command in enumerate(matrix(args.group)):
        if json.dumps(command) in completed and not args.retry:
            continue
        logfile = folder / f"{i:03d}.txt"
        print(command, flush=True)
        started = datetime.now().astimezone().isoformat()
        with logfile.open("w", encoding="utf-8") as log:
            proc = subprocess.run([sys.executable, "-u", *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  env=__import__("os").environ | {"PYTHONUTF8": "1"})
        history.append({"command": command, "exit_code": proc.returncode, "started": started, "log": str(logfile.relative_to(ROOT))})
        index.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print("exit", proc.returncode, flush=True)
    done = {json.dumps(h["command"]) for h in history if h["exit_code"] == 0}
    if all(json.dumps(command) in done for command in matrix(args.group)):
        (folder / "all-complete.json").write_text(json.dumps({"status": "completed", "runs": len(matrix(args.group))}), encoding="utf-8")
