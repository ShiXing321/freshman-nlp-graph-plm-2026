"""Small pretrained BERT experiments; keeps upstream examples intact."""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForMaskedLM, BertTokenizerFast

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
sys.path.insert(0, str(ROOT))
from common.experiment import Run, seed_all, device


def prepare_model():
    folder = ROOT / ".cache/models/bert-small-zh"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ["config.json", "vocab.txt", "pytorch_model.bin"]:
        target = folder / name
        if not target.exists():
            source = ROOT / (".cache/bert-" + name)
            target.write_bytes(source.read_bytes())
    tok = BertTokenizerFast(vocab_file=str(folder / "vocab.txt"), do_lower_case=True)
    tok.save_pretrained(folder)
    return folder, tok


def load_rows(task, seed):
    if task in ["classification", "pet", "ptuning"]:
        relative = "text_classification" if task == "classification" else ("prompt_tasks/PET" if task == "pet" else "prompt_tasks/p-tuning")
        folder = BASE / relative / "data/comment_classify"
        # Category names are schema metadata. PET dev includes an unseen category (电器).
        labels = sorted({line.split("\t", 1)[0] for s in ["train", "dev"] for line in (folder / f"{s}.txt").read_text(encoding="utf-8").splitlines() if line})
        mapping = {label: i for i, label in enumerate(labels)}
        def parse(line):
            label, text = line.split("\t", 1)
            return {"text": text, "label": mapping[label]}
        raw = {s: [parse(l) for l in (folder / f"{s}.txt").read_text(encoding="utf-8").splitlines() if l] for s in ["train", "dev"]}
    elif task == "simcse":
        folder = BASE / "text_matching/unsupervised/simcse/data/LCQMC"
        raw = {}
        for split in ["dev", "test"]:
            raw[split] = []
            skipped = 0
            for line in (folder / f"{split}.tsv").read_text(encoding="utf-8").splitlines():
                fields = line.split("\t")
                if len(fields) != 3 or fields[2] not in ["0", "1"]:
                    skipped += 1
                    continue
                raw[split].append({"a": fields[0], "b": fields[1], "label": int(fields[2])})
            print(split, "valid", len(raw[split]), "filtered", skipped, flush=True)
        if not raw["test"]:
            # Upstream test.tsv has no labels. Reserve sentence-connected dev groups.
            from sklearn.model_selection import GroupShuffleSplit
            parent = {}
            def find(t):
                parent.setdefault(t, t)
                if parent[t] != t:
                    parent[t] = find(parent[t])
                return parent[t]
            for r in raw["dev"]:
                parent[find(r["a"])] = find(r["b"])
            groups = [find(r["a"]) for r in raw["dev"]]
            left, right = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=seed).split(raw["dev"], groups=groups))
            labeled = raw["dev"]
            raw["dev"], raw["test"] = [labeled[i] for i in left], [labeled[i] for i in right]
            print("unlabeled upstream test: heldout evaluation uses sentence-connected groups from dev", flush=True)
        holdout = {r[k] for s in ["dev", "test"] for r in raw[s] for k in ["a", "b"]}
        texts = (folder / "train.txt").read_text(encoding="utf-8").splitlines()
        raw["train"] = [{"a": t, "b": t, "label": 0} for t in dict.fromkeys(texts) if t.strip() and t not in holdout]
        print("train", "original", len(texts), "deduplicated and heldout removed", len(raw["train"]), flush=True)
        return raw, ["different", "same"]
    else:
        folder = BASE / "text_matching/supervised/data/comment_classify"
        def parse(line):
            a, b, y = line.split("\t")
            return {"a": a, "b": b, "label": int(y)}
        raw = {s: [parse(l) for l in (folder / f"{s}.txt").read_text(encoding="utf-8").splitlines() if l] for s in ["train", "dev"]}
        labels = ["different", "same"]
    # Upstream often supplies no test set. Reserve half its dev set once, without tuning on it.
    from collections import Counter
    stratify = [r["label"] for r in raw["dev"]]
    if min(Counter(stratify).values()) < 2:
        stratify = None
    def key(row):
        return row.get("text", row.get("b", ""))
    if task in ["pointwise", "dssm", "sbert"]:
        from sklearn.model_selection import GroupShuffleSplit
        left, right = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=seed).split(raw["dev"], groups=[key(r) for r in raw["dev"]]))
        dev, test = [raw["dev"][i] for i in left], [raw["dev"][i] for i in right]
    else:
        dev, test = train_test_split(raw["dev"], test_size=.5, random_state=seed, stratify=stratify)
    holdout = {key(r) for r in dev + test}
    train = [r for r in raw["train"] if key(r) not in holdout]
    return {"train": train, "dev": dev, "test": test}, labels


class BertTask(nn.Module):
    def __init__(self, pretrained, tok, task, labels, prompt_length=8):
        super().__init__()
        self.task, self.tok = task, tok
        self.prompt_length = prompt_length
        if task in ["pet", "ptuning"]:
            self.encoder = AutoModelForMaskedLM.from_pretrained(pretrained, local_files_only=True)
            # Label words are two Chinese characters in the supplied category verbalizer.
            self.words = [tok.encode(label, add_special_tokens=False) for label in labels]
            lengths = {len(w) for w in self.words}
            if len(lengths) != 1:
                raise ValueError("This small experiment needs equal-length label words")
            self.mask_length = next(iter(lengths))
            if task == "ptuning":
                for p in self.encoder.parameters():
                    p.requires_grad_(False)
                dim = self.encoder.config.hidden_size
                self.prompt = nn.Parameter(torch.randn(prompt_length, dim) * .02)
        else:
            self.encoder = AutoModel.from_pretrained(pretrained, local_files_only=True)
            dim = self.encoder.config.hidden_size
            self.head = nn.Linear(dim * 3 if task == "sbert" else dim, len(labels))
            self.projection = nn.Linear(dim, dim)
            self.scale = nn.Parameter(torch.tensor(5.))
            self.bias = nn.Parameter(torch.tensor(0.))

    def encode(self, batch):
        output = self.encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        # Mean pooling excludes padding, CLS, and SEP.
        valid = mask * ((batch["input_ids"] != self.tok.cls_token_id) & (batch["input_ids"] != self.tok.sep_token_id)).unsqueeze(-1)
        return (output * valid).sum(1) / valid.sum(1).clamp_min(1)

    def prompt_inputs(self, texts, dev):
        prompts = ["这是一条" + self.tok.mask_token * self.mask_length + "评论：" + text[:90] for text in texts]
        batch = self.tok(prompts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(dev)
        positions = batch["input_ids"].eq(self.tok.mask_token_id).nonzero()[:, 1].reshape(len(texts), self.mask_length)
        if self.task == "ptuning":
            embeddings = self.encoder.get_input_embeddings()(batch["input_ids"])
            prompt = self.prompt.unsqueeze(0).expand(len(texts), -1, -1)
            embeddings = torch.cat([embeddings[:, :1], prompt, embeddings[:, 1:]], 1)
            attention = torch.cat([batch["attention_mask"][:, :1], torch.ones(len(texts), self.prompt_length, device=dev), batch["attention_mask"][:, 1:]], 1).long()
            output = self.encoder(inputs_embeds=embeddings, attention_mask=attention).logits
            positions = positions + self.prompt_length
        else:
            output = self.encoder(**batch).logits
        selected = output[torch.arange(len(texts), device=dev)[:, None], positions]
        logits = selected.log_softmax(-1)
        scores = torch.stack([sum(logits[:, i, token] for i, token in enumerate(word)) for word in self.words], -1)
        return scores, selected

    def forward(self, rows, dev):
        y = torch.tensor([r["label"] for r in rows], device=dev)
        def inputs(texts, pairs=None):
            return self.tok(texts, pairs, padding=True, truncation=True, max_length=128, return_tensors="pt").to(dev)
        if self.task in ["pet", "ptuning"]:
            scores, masked = self.prompt_inputs([r["text"] for r in rows], dev)
            target = torch.tensor([self.words[int(i)] for i in y], device=dev)
            loss = nn.functional.cross_entropy(masked.flatten(0, 1), target.flatten())
        elif self.task in ["classification", "pointwise"]:
            batch = inputs([r["text"] for r in rows]) if self.task == "classification" else inputs([r["a"] for r in rows], [r["b"] for r in rows])
            hidden = self.encoder(**batch).last_hidden_state[:, 0]
            scores = self.head(hidden)
            loss = nn.functional.cross_entropy(scores, y)
        else:
            u = self.encode(inputs([r["a"] for r in rows]))
            v = self.encode(inputs([r["a"] if self.task == "simcse" and self.training else r["b"] for r in rows]))
            if self.task == "sbert":
                scores = self.head(torch.cat([u, v, (u - v).abs()], -1))
                loss = nn.functional.cross_entropy(scores, y)
            elif self.task == "dssm":
                cosine = nn.functional.cosine_similarity(self.projection(u), self.projection(v))
                score = cosine * self.scale + self.bias
                scores = torch.stack([torch.zeros_like(score), score], -1)
                loss = nn.functional.binary_cross_entropy_with_logits(score, y.float())
            else:
                # Unsupervised SimCSE: independent dropout views of exactly the same sentence.
                # The second sentence and labels are reserved for evaluation only.
                if self.training:
                    logits = nn.functional.normalize(u, dim=-1) @ nn.functional.normalize(v, dim=-1).T / .05
                    labels = torch.arange(len(rows), device=dev)
                    loss = (nn.functional.cross_entropy(logits, labels) + nn.functional.cross_entropy(logits.T, labels)) / 2
                else:
                    loss = torch.tensor(0., device=dev)
                score = nn.functional.cosine_similarity(u, v)
                scores = torch.stack([torch.zeros_like(score), score], -1)
        return loss, scores


@torch.no_grad()
def evaluate(model, rows, dev, batch_size, threshold=None):
    model.eval()
    scores, truth = [], []
    for batch in DataLoader(rows, batch_size=batch_size, collate_fn=list):
        scores.extend(model(batch, dev)[1].cpu().tolist())
        truth.extend(r["label"] for r in batch)
    scores = np.array(scores)
    if model.task == "simcse":
        if threshold is None:
            threshold = max(np.linspace(-1, 1, 101), key=lambda t: f1_score(truth, scores[:, 1] >= t, zero_division=0))
        predictions = scores[:, 1] >= threshold
    else:
        predictions = scores.argmax(1)
    metrics = {"accuracy": accuracy_score(truth, predictions), "macro_f1": f1_score(truth, predictions, average="macro", zero_division=0)}
    if scores.shape[1] == 2:
        metrics.update(auc=roc_auc_score(truth, scores[:, 1] - scores[:, 0]), average_precision=average_precision_score(truth, scores[:, 1] - scores[:, 0]))
    return metrics, float(threshold) if threshold is not None else None


def main(args):
    seed_all(args.seed)
    pretrained, tok = prepare_model()
    rows, labels = load_rows(args.task, args.seed)
    if args.limit:
        rng = random.Random(args.seed)
        rows = {s: rng.sample(r, min(len(r), args.limit if s == "train" else 1000)) for s, r in rows.items()}
    dev = device()
    model = BertTask(pretrained, tok, args.task, labels).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    config = vars(args) | {"pretrained": "uer/chinese_roberta_L-2_H-128", "counts": {s: len(r) for s, r in rows.items()}, "labels": labels,
                           "test_split": "sentence-connected group split of labeled LCQMC dev; upstream test has no labels" if args.task == "simcse" else "half of upstream dev, seeded split; stratified if all classes have >=2 examples", "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if args.task in ["pointwise", "dssm", "sbert"]:
        config["test_split"] = "grouped half of upstream dev by comment text; train holdout overlaps removed"
    run = Run(BASE / "local_runs/results", f"{args.task}_n{args.limit}_lr{args.lr}" + ("_grouped" if args.task in ["pointwise", "dssm", "sbert"] else ""), config)
    best, state, threshold = -1, None, None
    for epoch in range(args.epochs):
        model.train()
        total, count = 0., 0
        for batch in DataLoader(rows["train"], batch_size=args.batch, shuffle=True, collate_fn=list):
            if args.task == "simcse" and len(batch) < 2:
                continue
            opt.zero_grad()
            loss = model(batch, dev)[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            total += float(loss.detach()) * len(batch)
            count += len(batch)
        metrics, candidate_threshold = evaluate(model, rows["dev"], dev, args.batch)
        run.epoch(epoch=epoch + 1, loss=total / count, dev=metrics)
        if metrics["macro_f1"] > best:
            best, state, threshold = metrics["macro_f1"], copy.deepcopy(model.state_dict()), candidate_threshold
    model.load_state_dict(state)
    metrics, _ = evaluate(model, rows["test"], dev, args.batch, threshold)
    ckpt = BASE / "local_runs/checkpoints" / args.task
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.cpu().state_dict(), "config": config, "threshold": threshold}, ckpt / "model.pt")
    tok.save_pretrained(ckpt)
    run.finish(best_dev_macro_f1=best, test=metrics, threshold=threshold)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["classification", "pointwise", "dssm", "sbert", "simcse", "pet", "ptuning"])
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
