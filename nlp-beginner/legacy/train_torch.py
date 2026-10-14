"""Tasks 2--5. Example: python train_torch.py classification --model cnn."""
import argparse
import copy
import json
import math
import re
import sys
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.experiment import Run, device, seed_all
from models import TextClassifier, ESIM, BiLSTMCRF, CharLM

BASE = Path(__file__).resolve().parent
TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())


def read_rows(name, limit, seed, eval_limit=1000):
    result = {}
    rng = np.random.default_rng(seed)
    for split in ["train", "dev", "test"]:
        path = BASE / "data/processed" / name / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        n = limit if split == "train" else (min(eval_limit, limit) if limit and eval_limit else 0)
        if n and len(rows) > n:
            rows = [rows[i] for i in sorted(rng.choice(len(rows), n, replace=False))]
        result[split] = rows
    return result


def word_case(word):
    if word.isdigit():
        return 1
    if word.islower():
        return 2
    if word.isupper():
        return 3
    if word.istitle():
        return 4
    if any(c.isdigit() for c in word):
        return 5
    if any(c.isalpha() for c in word):
        return 6
    return 7


def collate(batch, task, vocab, max_length, use_case=False):
    def encode(tokens):
        return torch.tensor([vocab.get(t, 1) for t in tokens[:max_length]] or [1])
    if task == "classification":
        return (pad_sequence([encode(tokenize(r["text"])) for r in batch], batch_first=True), torch.tensor([r["label"] for r in batch]))
    if task == "matching":
        return (pad_sequence([encode(tokenize(r["premise"])) for r in batch], batch_first=True),
                pad_sequence([encode(tokenize(r["hypothesis"])) for r in batch], batch_first=True), torch.tensor([r["label"] for r in batch]))
    ids = pad_sequence([encode([t.lower() for t in r["tokens"]]) for r in batch], batch_first=True)
    labels = pad_sequence([torch.tensor(r["tags"][:max_length]) for r in batch], batch_first=True)
    if use_case:
        cases = pad_sequence([torch.tensor([word_case(t) for t in r["tokens"][:max_length]]) for r in batch], batch_first=True)
        return ids, cases, labels
    return ids, labels


def glove_vectors(path, vocab, dim):
    matrix = torch.randn(len(vocab), dim) * .1
    matrix[0].zero_()
    hits = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            word, _, values = line.partition(" ")
            if word in vocab:
                vector = np.fromstring(values, sep=" ", dtype=np.float32)
                if len(vector) == dim:
                    matrix[vocab[word]] = torch.tensor(vector)
                    hits += 1
    return matrix, hits


@torch.no_grad()
def evaluate(model, loader, task, dev):
    model.eval()
    truth, predictions = [], []
    for batch in loader:
        batch = [x.to(dev) for x in batch]
        output = model(*batch[:-1])
        if task == "ner":
            paths = model.crf.decode(output, batch[0].ne(0))
            for i, path in enumerate(paths):
                truth.append([TAGS[int(t)] for t in batch[-1][i, :len(path)]])
                predictions.append([TAGS[t] for t in path])
        else:
            truth.extend(batch[-1].cpu().tolist())
            predictions.extend(output.argmax(-1).cpu().tolist())
    if task == "ner":
        from seqeval.metrics import precision_score, recall_score, f1_score as entity_f1
        return {"precision": precision_score(truth, predictions, zero_division=0),
                "recall": recall_score(truth, predictions, zero_division=0), "f1": entity_f1(truth, predictions, zero_division=0)}
    return {"accuracy": accuracy_score(truth, predictions), "macro_f1": f1_score(truth, predictions, average="macro", zero_division=0)}


def supervised(args):
    name = {"classification": args.data_name, "matching": "snli", "ner": "ner"}[args.task]
    rows = read_rows(name, args.limit, args.seed, args.eval_limit)
    counter = Counter()
    for row in rows["train"]:
        if args.task == "classification":
            counter.update(tokenize(row["text"]))
        elif args.task == "matching":
            counter.update(tokenize(row["premise"]) + tokenize(row["hypothesis"]))
        else:
            counter.update(t.lower() for t in row["tokens"])
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, count in counter.most_common(30000):
        if word not in vocab:
            vocab[word] = len(vocab)
    loaders = {s: DataLoader(r, batch_size=args.batch, shuffle=s == "train", collate_fn=partial(collate, task=args.task, vocab=vocab, max_length=args.max_length, use_case=args.use_case), num_workers=0) for s, r in rows.items()}
    coverage = None
    if args.task == "classification":
        vectors = None
        if args.glove:
            vectors, hits = glove_vectors(args.glove, vocab, args.dim)
            coverage = hits / len(vocab)
        seed_all(args.seed)
        model = TextClassifier(len(vocab), args.model, args.dim, args.hidden, vectors=vectors)
    elif args.task == "matching":
        model = ESIM(len(vocab), args.dim, args.hidden)
    else:
        model = BiLSTMCRF(len(vocab), dim=args.dim, hidden=args.hidden, use_case=args.use_case)
    if args.task != "classification" and args.glove:
        vectors, hits = glove_vectors(args.glove, vocab, args.dim)
        model.embedding.weight.data.copy_(vectors)
        coverage = hits / len(vocab)
    dev = device()
    model.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    folder = BASE / {"classification": "task2", "matching": "task3", "ner": "task4"}[args.task]
    suffix = "_kaggle" if name == "sentiment_kaggle" else ""
    suffix += args.tag
    run = Run(folder / "results", f"{args.model}_{'glove' if args.glove else 'random'}_n{args.limit}_lr{args.lr}" + suffix, vars(args))
    best, state = -1, None
    for epoch in range(args.epochs):
        model.train()
        total, count = 0., 0
        for batch in loaders["train"]:
            batch = [x.to(dev) for x in batch]
            optimizer.zero_grad()
            output = model(*batch[:-1])
            loss = model.crf(output, batch[-1], batch[0].ne(0)) if args.task == "ner" else torch.nn.functional.cross_entropy(output, batch[-1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.detach()) * batch[0].size(0)
            count += batch[0].size(0)
        metrics = evaluate(model, loaders["dev"], args.task, dev)
        run.epoch(epoch=epoch + 1, train_loss=total / count, dev=metrics)
        score = metrics["f1"] if args.task == "ner" else metrics["accuracy"]
        if score > best:
            best = score
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    (folder / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save({"state": state, "vocab": vocab, "config": vars(args)}, folder / "checkpoints" / (run.data["name"] + ".pt"))
    run.finish(best_dev_score=best, test=evaluate(model, loaders["test"], args.task, dev), counts={s: len(r) for s, r in rows.items()}, glove_coverage=coverage)


def language_model(args):
    rows = read_rows("poetry", 0, args.seed)
    texts = {s: r[0]["text"] for s, r in rows.items()}
    vocab = {"<unk>": 0}
    for char in sorted(set(texts["train"])):
        vocab[char] = len(vocab)
    reverse = {v: k for k, v in vocab.items()}
    streams = {s: torch.tensor([vocab.get(c, 0) for c in text]) for s, text in texts.items()}
    windows = {s: [(ids[i:i + args.max_length], ids[i + 1:i + args.max_length + 1]) for i in range(0, len(ids) - args.max_length, args.max_length)] for s, ids in streams.items()}
    loaders = {s: DataLoader(items, batch_size=args.batch, shuffle=s == "train") for s, items in windows.items()}
    dev = device()
    model = CharLM(len(vocab), args.model, args.dim, args.hidden).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    folder = BASE / "task5"
    run = Run(folder / "results", f"{args.model}_lr{args.lr}", vars(args))
    def epoch_loss(split, train=False):
        model.train(train)
        total, count = 0., 0
        with torch.set_grad_enabled(train):
            for x, y in loaders[split]:
                x, y = x.to(dev), y.to(dev)
                logits = model(x)[0]
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, len(vocab)), y.reshape(-1))
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                total += float(loss.detach()) * y.numel()
                count += y.numel()
        return total / count
    best, state = math.inf, None
    for epoch in range(args.epochs):
        train_loss, dev_loss = epoch_loss("train", True), epoch_loss("dev")
        run.epoch(epoch=epoch + 1, train_loss=train_loss, dev_loss=dev_loss, dev_perplexity=math.exp(dev_loss))
        if dev_loss < best:
            best, state = dev_loss, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    test_loss = epoch_loss("test")
    model.eval()
    outputs = {}
    with torch.no_grad():
        for temperature in [.7, 1.0]:
            ids = torch.tensor([[vocab.get(c, 0) for c in "春风"]], device=dev)
            state = None
            output = "春风"
            for step in range(80):
                logits, state = model(ids, state)
                ids = torch.multinomial((logits[:, -1] / temperature).softmax(-1), 1)
                output += reverse[int(ids.item())]
            outputs[str(temperature)] = output
    (folder / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.cpu().state_dict(), "vocab": vocab, "config": vars(args)}, folder / "checkpoints" / (run.data["name"] + ".pt"))
    run.finish(test_loss=test_loss, test_perplexity=math.exp(test_loss), generated=outputs,
               character_counts={s: len(t) for s, t in texts.items()}, unseen_test_characters=sum(c not in vocab for c in texts["test"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["classification", "matching", "ner", "lm"])
    p.add_argument("--data-name", default="sentiment", choices=["sentiment", "sentiment_kaggle"])
    p.add_argument("--eval-limit", type=int, default=1000, help="0 evaluates full validation and test splits")
    p.add_argument("--tag", default="")
    p.add_argument("--use-case", action="store_true", help="Append 8-dimensional word casing features for NER")
    p.add_argument("--model", default="cnn")
    p.add_argument("--glove", default="")
    p.add_argument("--dim", type=int, default=50)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--max-length", type=int, default=100)
    p.add_argument("--limit", type=int, default=0, help="0 uses all examples; positive uses a seeded subset")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    seed_all(args.seed)
    language_model(args) if args.task == "lm" else supervised(args)
