"""Exact PyTorch rewrite of the NumPy TF-IDF softmax/OVR logistic baseline."""
import argparse
import copy
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.experiment import Run, device, seed_all
from train_numpy import objective

BASE = Path(__file__).resolve().parent

def main(args):
    seed_all(args.seed)
    rows = {s: [json.loads(l) for l in (BASE / "data/processed" / args.data_name / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()] for s in ["train", "dev", "test"]}
    vectorizer = TfidfVectorizer(min_df=2, max_features=15000, dtype=np.float32)
    x = {"train": vectorizer.fit_transform([r["text"] for r in rows["train"]])}
    x.update({s: vectorizer.transform([r["text"] for r in rows[s]]) for s in ["dev", "test"]})
    y = {s: np.array([r["label"] for r in rows[s]]) for s in rows}
    model = torch.nn.Linear(x["train"].shape[1], 5).to(device())
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    opt = torch.optim.SGD([{"params": model.weight, "weight_decay": 1e-4}, {"params": model.bias, "weight_decay": 0.}], lr=args.lr)
    def loss_function(scores, target):
        if args.loss == "softmax":
            return torch.nn.functional.cross_entropy(scores, target)
        return torch.nn.functional.binary_cross_entropy_with_logits(scores, torch.nn.functional.one_hot(target, 5).float(), reduction="none").sum(-1).mean()
    # Compare loss and gradients with independent NumPy code on a real data mini-batch.
    matrix, targets = x["train"][:8], y["train"][:8]
    tensor = torch.tensor(matrix.toarray(), device=device())
    loss = loss_function(model(tensor), torch.tensor(targets, dtype=torch.long, device=device()))
    loss.backward()
    numpy_loss, dw, db = objective(matrix, targets, np.zeros((matrix.shape[1], 5), np.float32), np.zeros(5, np.float32), args.loss, regularization=0.)
    assert abs(float(loss.detach()) - numpy_loss) < 1e-6
    assert np.max(np.abs(model.weight.grad.cpu().numpy().T - dw)) < 1e-6
    assert np.max(np.abs(model.bias.grad.cpu().numpy() - db)) < 1e-6
    opt.zero_grad()
    suffix = "_kaggle" if args.data_name == "sentiment_kaggle" else ""
    run = Run(BASE / "task2/results", "torch_bow_" + args.loss + suffix, vars(args) | {"features": "TF-IDF unigram; identical preprocessing and zero initialization to task1", "gradient_equivalence": "NumPy and PyTorch checked at zero weights", "counts": {s: len(r) for s, r in rows.items()}})
    @torch.no_grad()
    def evaluate(split):
        pred = []
        for start in range(0, len(y[split]), args.batch):
            logits = model(torch.tensor(x[split][start:start+args.batch].toarray(), device=device()))
            pred.extend(logits.argmax(-1).cpu().tolist())
        return {"accuracy": accuracy_score(y[split], pred), "macro_f1": f1_score(y[split], pred, average="macro", zero_division=0)}
    best, state = -1, None
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(y["train"]))
        total, count = 0., 0
        for start in range(0, len(order), args.batch):
            idx = order[start:start+args.batch]
            opt.zero_grad()
            loss = loss_function(model(torch.tensor(x["train"][idx].toarray(), device=device())), torch.tensor(y["train"][idx], dtype=torch.long, device=device()))
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
            count += len(idx)
        metrics = evaluate("dev")
        run.epoch(epoch=epoch+1, train_loss=total/count, dev=metrics)
        if metrics["accuracy"] > best:
            best, state = metrics["accuracy"], copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    folder = BASE / "task2/checkpoints"
    folder.mkdir(exist_ok=True, parents=True)
    torch.save({"state": model.cpu().state_dict(), "config": vars(args)}, folder / (run.data["name"] + ".pt"))
    joblib.dump(vectorizer, folder / (run.data["name"] + "_vectorizer.joblib"))
    model.to(device())
    run.finish(best_dev_accuracy=best, test=evaluate("test"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-name", default="sentiment", choices=["sentiment", "sentiment_kaggle"])
    p.add_argument("--loss", choices=["softmax", "logistic"], default="softmax")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=.5)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
