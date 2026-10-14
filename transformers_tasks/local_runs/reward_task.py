"""Bradley--Terry preference learning on the repository's ranked Chinese reviews."""
import argparse
import copy
import json
import random
import re
import sys
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel
from bert_tasks import prepare_model

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE.parent))
from common.experiment import Run, device, seed_all


def rows(limit, seed):
    folder = BASE / "RLHF/data/reward_datasets/sentiment_analysis"
    def parse(path):
        result = []
        for line in path.read_text(encoding="utf-8").splitlines():
            rank = [re.sub(r"^\s*\d+[.、]\s*", "", t) for t in line.split("\t")]
            if len(rank) >= 2:
                result.append(rank)
        return result
    train, dev = parse(folder / "train.tsv"), parse(folder / "dev.tsv")
    rng = random.Random(seed)
    rng.shuffle(dev)
    val, test = dev[:len(dev) // 2], dev[len(dev) // 2:]
    holdout = {text for rank in dev for text in rank}
    train = [rank for rank in train if all(text not in holdout for text in rank)]
    if limit:
        train = rng.sample(train, min(limit, len(train)))
        val, test = val[:200], test[:200]
    def pairs(ranks):
        return [(rank[i], rank[j]) for rank in ranks for i in range(len(rank)) for j in range(i + 1, len(rank))]
    return {"train": pairs(train), "dev": pairs(val), "test": pairs(test)}


class RewardModel(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained, local_files_only=True)
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, inputs):
        return self.head(self.encoder(**inputs).last_hidden_state[:, 0]).flatten()


def main(args):
    seed_all(args.seed)
    pretrained, tok = prepare_model()
    data = rows(args.limit, args.seed)
    dev = device()
    model = RewardModel(pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    run = Run(BASE / "local_runs/results", f"reward_n{args.limit}", vars(args) | {"counts": {s: len(r) for s, r in data.items()}, "loss": "-log sigmoid(r(chosen)-r(rejected))"})
    def score(texts):
        return model(tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(dev))
    @torch.no_grad()
    def evaluate(split):
        model.eval()
        correct, count = 0, 0
        for batch in DataLoader(data[split], batch_size=args.batch, collate_fn=list):
            chosen, rejected = zip(*batch)
            correct += int((score(list(chosen)) > score(list(rejected))).sum())
            count += len(batch)
        return {"preference_accuracy": correct / count, "pairs": count}
    best, state = -1, None
    for epoch in range(args.epochs):
        model.train()
        total, count = 0., 0
        for batch in DataLoader(data["train"], batch_size=args.batch, shuffle=True, collate_fn=list):
            chosen, rejected = zip(*batch)
            opt.zero_grad()
            loss = nn.functional.softplus(-(score(list(chosen)) - score(list(rejected)))).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            total += float(loss.detach()) * len(batch)
            count += len(batch)
        metrics = evaluate("dev")
        run.epoch(epoch=epoch + 1, loss=total / count, dev=metrics)
        if metrics["preference_accuracy"] > best:
            best, state = metrics["preference_accuracy"], copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    metrics = evaluate("test")
    examples = ["买过很多箱这个苹果了，一如既往的好，汁多味甜。", "一台充电很慢，信号不好，退了又买一台竟然是次品。"]
    model.eval()
    with torch.no_grad():
        scores = score(examples).cpu().tolist()
    ckpt = BASE / "local_runs/checkpoints/reward"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.cpu().state_dict(), "config": vars(args)}, ckpt / "model.pt")
    run.finish(best_dev_preference_accuracy=best, test=metrics, examples=[{"text": t, "score": s} for t, s in zip(examples, scores)])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
