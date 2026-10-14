"""Chinese T5 question answering and masked-span filling, with held-out generation."""
import argparse
import copy
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, T5ForConditionalGeneration

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
sys.path.insert(0, str(ROOT))
from common.experiment import Run, device, seed_all
PARSE_STATS = {}


def load_data(task, seed, limit, eval_limit=100):
    rng = random.Random(seed)
    if task == "qa":
        folder = BASE / "answer_generation/data/DuReaderQG"
        def parse(path):
            rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]
            return [{"input": "问题：" + r["question"] + " 原文：" + r["context"], "target": r["answer"], "group": r["context"]} for r in rows]
        train, dev = parse(folder / "train.json"), parse(folder / "dev.json")
    else:
        folder = BASE / "data_augment/filling_model/data"
        def parse(path):
            result = []
            skipped = 0
            for l in path.read_text(encoding="utf-8").splitlines():
                if l:
                    if "\t" not in l:
                        skipped += 1
                        continue
                    source, target = l.split("\t", 1)
                    if not source.strip() or not target.strip():
                        skipped += 1
                        continue
                    result.append({"input": source, "target": target, "group": target})
            PARSE_STATS[path.name] = {"valid": len(result), "malformed_filtered": skipped}
            print(path.name, PARSE_STATS[path.name], flush=True)
            return result
        train, dev = parse(folder / "train.tsv"), parse(folder / "dev.tsv")
    groups = sorted({r["group"] for r in dev})
    rng.shuffle(groups)
    test_groups = set(groups[:max(1, len(groups) // 2)])
    holdout = set(groups)
    rows = {"train": [r for r in train if r["group"] not in holdout],
            "dev": [r for r in dev if r["group"] not in test_groups], "test": [r for r in dev if r["group"] in test_groups]}
    if limit:
        rows = {s: rng.sample(r, min(len(r), limit if s == "train" else eval_limit)) for s, r in rows.items()}
    return rows


def normalized(text):
    return re.sub(r"\s+", "", text).replace("[SEP]", "").replace("[CLS]", "")


def character_f1(prediction, target):
    a, b = Counter(normalized(prediction)), Counter(normalized(target))
    overlap = sum((a & b).values())
    return 2 * overlap / max(sum(a.values()) + sum(b.values()), 1)


def main(args):
    seed_all(args.seed)
    folder = ROOT / ".cache/models/t5-small-zh"
    tok = BertTokenizerFast(vocab_file=str(folder / "vocab.txt"), do_lower_case=False)
    tok.bos_token, tok.eos_token = tok.cls_token, tok.sep_token
    model = T5ForConditionalGeneration.from_pretrained(folder, local_files_only=True)
    model.config.eos_token_id = tok.sep_token_id
    model.generation_config.eos_token_id = tok.sep_token_id
    dev = device()
    model.to(dev)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    rows = load_data(args.task, args.seed, args.limit, args.eval_limit)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    run = Run(BASE / "local_runs/results", f"t5_{args.task}_n{args.limit}_v2" + args.tag, vars(args) | {"pretrained": "uer/t5-small-chinese-cluecorpussmall", "counts": {s: len(r) for s, r in rows.items()}, "parsing": PARSE_STATS, "labels": "target tokens + SEP, no CLS in labels; decoder_start_token=CLS, padding=-100", "split": "grouped held-out half of upstream dev, overlapping training groups removed"})
    def encode(batch):
        inputs = tok([r["input"] for r in batch], padding=True, truncation=True, max_length=args.source_length, return_tensors="pt").to(dev)
        inputs.pop("token_type_ids", None)
        # Decoder start already supplies CLS. Including it in targets repeats BOS.
        ids = [tok.encode(r["target"], add_special_tokens=False)[:args.target_length - 1] + [tok.sep_token_id] for r in batch]
        width = max(len(x) for x in ids)
        labels = torch.tensor([x + [-100] * (width - len(x)) for x in ids], device=dev)
        return inputs, labels
    @torch.no_grad()
    def evaluate(split, generate=False):
        model.eval()
        total, tokens = 0., 0
        em, f1, count = 0., 0., 0
        examples = []
        for batch in DataLoader(rows[split], batch_size=args.batch, collate_fn=list):
            inputs, labels = encode(batch)
            loss = model(**inputs, labels=labels).loss
            n = int(labels.ne(-100).sum())
            total, tokens = total + float(loss) * n, tokens + n
            if generate:
                output = model.generate(**inputs, max_new_tokens=args.target_length, do_sample=False, num_beams=1, use_cache=True)
                predictions = tok.batch_decode(output, skip_special_tokens=True)
                for row, prediction in zip(batch, predictions):
                    em += normalized(prediction) == normalized(row["target"])
                    f1 += character_f1(prediction, row["target"])
                    count += 1
                    if len(examples) < 10:
                        examples.append({"input": row["input"], "gold": row["target"], "prediction": prediction})
        metrics = {"loss": total / max(tokens, 1)}
        if generate:
            metrics.update(exact_match=em / count, character_f1=f1 / count, examples=examples)
        return metrics
    best, state = float("inf"), None
    for epoch in range(args.epochs):
        model.train()
        total, count = 0., 0
        for batch in DataLoader(rows["train"], batch_size=args.batch, shuffle=True, collate_fn=list):
            inputs, labels = encode(batch)
            opt.zero_grad()
            loss = model(**inputs, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            total += float(loss.detach()) * len(batch)
            count += len(batch)
        metrics = evaluate("dev")
        run.epoch(epoch=epoch + 1, loss=total / count, dev=metrics)
        if metrics["loss"] < best:
            best = metrics["loss"]
            # CPU snapshots avoid doubling the weights in limited GPU memory.
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    metrics = evaluate("test", generate=True)
    ckpt = BASE / "local_runs/checkpoints" / ("t5_" + args.task + args.tag)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    run.finish(best_dev_loss=best, test=metrics)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["qa", "filling"])
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--source-length", type=int, default=192)
    p.add_argument("--target-length", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-limit", type=int, default=100)
    p.add_argument("--tag", default="")
    main(p.parse_args())
