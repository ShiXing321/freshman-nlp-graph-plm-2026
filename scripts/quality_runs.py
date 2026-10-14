"""Run the submission experiments in sequence, with restartable logs."""
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from run_resume import completed_artifacts

ROOT = Path(__file__).resolve().parents[1]
NLP = "nlp-beginner/legacy/"
GLOVE = NLP + "data/raw/glove/glove.6B.50d.txt"


def jobs():
    for features, ngram, loss, lr, batch, shuffle in [
        ("count", 1, "softmax", .1, 64, True), ("binary", 1, "softmax", .1, 64, True),
        ("tfidf", 1, "softmax", .5, 64, True), ("tfidf", 2, "softmax", .5, 64, True),
        ("tfidf", 1, "logistic", .5, 64, True), ("tfidf", 1, "softmax", .05, 64, True),
        ("tfidf", 1, "softmax", .5, 0, True), ("tfidf", 1, "softmax", .05, 1, True),
        ("tfidf", 1, "softmax", .5, 64, False)]:
        yield [NLP + "train_numpy.py", "--data-name", "sentiment_kaggle", "--features", features,
               "--ngram", str(ngram), "--loss", loss, "--lr", str(lr), "--batch", str(batch),
               "--epochs", "20"] + ([] if shuffle else ["--no-shuffle"])
    for loss in ["softmax", "logistic"]:
        yield [NLP + "train_torch_bow.py", "--data-name", "sentiment_kaggle", "--loss", loss]
    for model, glove in [("softmax", False), ("cnn", False), ("rnn", False), ("cnn", True), ("rnn", True)]:
        yield [NLP + "train_torch.py", "classification", "--data-name", "sentiment_kaggle", "--model", model,
               "--epochs", "8", "--batch", "64"] + (["--glove", GLOVE] if glove else [])
    yield [NLP + "train_torch.py", "matching", "--model", "esim", "--limit", "100000",
           "--eval-limit", "0", "--glove", GLOVE, "--epochs", "10", "--batch", "64", "--tag", "_expanded"]
    yield [NLP + "train_torch.py", "ner", "--model", "bilstm-crf", "--glove", GLOVE,
           "--epochs", "20", "--max-length", "512", "--batch", "32", "--tag", "_expanded"]
    for task, epochs in [("task1", "100"), ("task2", "50")]:
        for ds, model, mode in itertools.product(["Cora", "CiteSeer", "Flickr"], ["gcn", "gat", "sage", "gin"], ["full", "sampled"]):
            yield [f"graph_beginner/{task}/code/train.py", "--dataset", ds, "--model", model,
                   "--mode", mode, "--epochs", epochs, "--hidden", "16" if ds == "Flickr" else "32", "--tag", "_isolated"]
    for seed, mode in itertools.product([7, 2026], ["full", "sampled"]):
        yield ["graph_beginner/task1/code/train.py", "--dataset", "Cora", "--model", "gcn",
               "--mode", mode, "--epochs", "100", "--seed", str(seed), "--tag", f"_seed{seed}"]
    for model in ["transe", "rotate", "conve"]:
        yield ["graph_beginner/task4/code/train.py", "--model", model, "--epochs", "20", "--limit", "0",
               "--eval-limit", "200", "--test-limit", "0", "--batch", "1024", "--negatives", "8", "--tag", "_full"]


def main():
    folder = ROOT / "运行记录/quality"
    folder.mkdir(parents=True, exist_ok=True)
    index = folder / "history.jsonl"
    history = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()] if index.exists() else []
    done = {json.dumps(row["command"]) for row in history if row["exit_code"] == 0}
    failures = []
    for i, command in enumerate(jobs()):
        if json.dumps(command) in done and completed_artifacts(command):
            continue
        print(datetime.now().astimezone().isoformat(), i, command, flush=True)
        start = datetime.now().astimezone().isoformat()
        with (folder / f"{i:03d}.txt").open("w", encoding="utf-8") as log:
            process = subprocess.run([sys.executable, "-X", "utf8", "-u", *command], cwd=ROOT,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     env=os.environ | {"PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1"})
        with index.open("a", encoding="utf-8") as out:
            out.write(json.dumps({"started": start, "finished": datetime.now().astimezone().isoformat(),
                                  "command": command, "exit_code": process.returncode, "log": f"{i:03d}.txt"}, ensure_ascii=False) + "\n")
        if process.returncode:
            failures.append(command)
        print(i, "exit", process.returncode, flush=True)
    (folder / "state.json").write_text(json.dumps({"status": "completed" if not failures else "failed", "failures": failures}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
