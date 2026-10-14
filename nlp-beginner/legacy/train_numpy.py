"""Task 1: sparse text features with NumPy softmax / one-vs-rest logistic SGD."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.experiment import Run, seed_all

BASE = Path(__file__).resolve().parent


def objective(x, y, w, b, loss, regularization=1e-4):
    logits = np.asarray(x @ w) + b
    target = np.eye(w.shape[1])[y]
    if loss == "softmax":
        shifted = logits - logits.max(1, keepdims=True)
        prob = np.exp(shifted)
        prob /= prob.sum(1, keepdims=True)
        value = -np.log(prob[np.arange(len(y)), y] + 1e-12).mean()
    else:
        prob = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
        value = (np.logaddexp(0, logits) - target * logits).sum(1).mean()
    error = (prob - target) / len(y)
    return float(value + .5 * regularization * (w * w).sum()), np.asarray(x.T @ error) + regularization * w, error.sum(0)


def main(args):
    seed_all(args.seed)
    rows = {s: [json.loads(line) for line in (BASE / "data/processed" / args.data_name / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()] for s in ["train", "dev", "test"]}
    cls = TfidfVectorizer if args.features == "tfidf" else CountVectorizer
    vec = cls(ngram_range=(1, args.ngram), min_df=2, max_features=15000, dtype=np.float32)
    x = {"train": vec.fit_transform([r["text"] for r in rows["train"]])}
    x.update({s: vec.transform([r["text"] for r in rows[s]]) for s in ["dev", "test"]})
    if args.features == "binary":
        for matrix in x.values():
            matrix.data.fill(1)
    y = {s: np.array([r["label"] for r in rows[s]]) for s in rows}
    w, b = np.zeros((x["train"].shape[1], 5), dtype=np.float32), np.zeros(5, dtype=np.float32)
    suffix = "_kaggle" if args.data_name == "sentiment_kaggle" else ""
    run = Run(BASE / "task1/results", f"{args.features}_n{args.ngram}_{args.loss}_lr{args.lr}_b{args.batch}_shuffle{args.shuffle}" + suffix, vars(args) | {"device_used": "cpu", "implementation": "NumPy"})
    n = len(y["train"])
    best, weights = -1, None
    rng = np.random.default_rng(args.seed)
    batch_size = args.batch if args.batch else n
    for epoch in range(args.epochs):
        order = rng.permutation(n) if args.shuffle else np.arange(n)
        losses = []
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            value, dw, db = objective(x["train"][idx], y["train"][idx], w, b, args.loss)
            w -= args.lr * dw
            b -= args.lr * db
            losses.append(value)
        pred = (x["dev"] @ w + b).argmax(1)
        acc = accuracy_score(y["dev"], pred)
        run.epoch(epoch=epoch + 1, train_loss=float(np.mean(losses)), dev_accuracy=acc)
        if acc > best:
            best, weights = acc, (w.copy(), b.copy())
    w, b = weights
    pred = (x["test"] @ w + b).argmax(1)
    ckpt = BASE / "task1/checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    np.savez(ckpt / (run.data["name"] + ".npz"), weights=w, bias=b, vocabulary=json.dumps({k: int(v) for k, v in vec.vocabulary_.items()}))
    import joblib
    joblib.dump(vec, ckpt / (run.data["name"] + "_vectorizer.joblib"))
    run.finish(best_dev_accuracy=best, test_accuracy=accuracy_score(y["test"], pred),
               test_macro_f1=f1_score(y["test"], pred, average="macro"), counts={s: len(rows[s]) for s in rows}, vocabulary_size=w.shape[0])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-name", default="sentiment", choices=["sentiment", "sentiment_kaggle"])
    p.add_argument("--features", choices=["count", "binary", "tfidf"], default="tfidf")
    p.add_argument("--ngram", type=int, default=1)
    p.add_argument("--loss", choices=["softmax", "logistic"], default="softmax")
    p.add_argument("--lr", type=float, default=.5)
    p.add_argument("--batch", type=int, default=64, help="0 means full batch; 1 means SGD")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    main(p.parse_args())
