"""UIE-style prompt-conditioned span extraction with a small BERT encoder."""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, BertTokenizerFast, T5ForConditionalGeneration
from safetensors.torch import load_file
from bert_tasks import prepare_model

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
sys.path.insert(0, str(BASE.parent))
from common.experiment import Run, device, seed_all


def data(seed):
    folder = BASE / "UIE/data/DuIE"
    rows = {s: [json.loads(l) for l in (folder / f"{s}.txt").read_text(encoding="utf-8").splitlines() if l] for s in ["train", "dev"]}
    # Rows with different prompts but identical content must stay together.
    contents = sorted({r["content"] for r in rows["dev"]})
    random.Random(seed).shuffle(contents)
    test_contents = set(contents[:max(1, len(contents) // 2)])
    holdout_contents = set(contents)
    return {"train": [r for r in rows["train"] if r["content"] not in holdout_contents],
            "dev": [r for r in rows["dev"] if r["content"] not in test_contents],
            "test": [r for r in rows["dev"] if r["content"] in test_contents]}


def encode(tok, rows, dev):
    encoded = tok([r["prompt"] for r in rows], [r["content"] for r in rows], padding=True, truncation=True, max_length=256, return_offsets_mapping=True, return_tensors="pt")
    offsets = encoded.pop("offset_mapping")
    valid = torch.tensor([[sid == 1 and int(offsets[i, j, 1]) > int(offsets[i, j, 0]) for j, sid in enumerate(encoded.sequence_ids(i))] for i in range(len(rows))])
    start, end = torch.zeros_like(valid, dtype=torch.float), torch.zeros_like(valid, dtype=torch.float)
    for i, row in enumerate(rows):
        for entity in row["result_list"]:
            starts = [j for j, (a, b) in enumerate(offsets[i].tolist()) if valid[i, j] and a <= entity["start"] < b]
            ends = [j for j, (a, b) in enumerate(offsets[i].tolist()) if valid[i, j] and a < entity["end"] <= b]
            if starts and ends:
                start[i, starts[0]], end[i, ends[0]] = 1, 1
    return encoded.to(dev), offsets, valid.to(dev), start.to(dev), end.to(dev)


class SpanModel(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained, local_files_only=True)
        self.head = nn.Linear(self.encoder.config.hidden_size, 2)
        boundary = Path(pretrained) / "boundary.safetensors"
        if boundary.exists():
            tensors = load_file(str(boundary))
            with torch.no_grad():
                self.head.weight.copy_(torch.cat([tensors["linear_start.weight"], tensors["linear_end.weight"]]))
                self.head.bias.copy_(torch.cat([tensors["linear_start.bias"], tensors["linear_end.bias"]]))
            self.encoder.gradient_checkpointing_enable()

    def forward(self, inputs):
        return self.head(self.encoder(**inputs).last_hidden_state)


def spans(prob, offsets, valid, threshold):
    starts = (prob[:, 0] >= threshold).nonzero().flatten().tolist()
    ends = (prob[:, 1] >= threshold).nonzero().flatten().tolist()
    result = set()
    for a in starts:
        if not valid[a]:
            continue
        candidates = [b for b in ends if a <= b and b - a < 40 and valid[b]]
        if candidates:
            b = candidates[0]
            result.add((int(offsets[a, 0]), int(offsets[b, 1])))
    return result


@torch.no_grad()
def evaluate(model, tok, rows, dev, batch_size, threshold=.5):
    model.eval()
    tp, predicted, gold = 0, 0, 0
    examples = []
    for batch in DataLoader(rows, batch_size=batch_size, collate_fn=list):
        inputs, offsets, valid, _, _ = encode(tok, batch, dev)
        probabilities = model(inputs).sigmoid().cpu()
        for i, row in enumerate(batch):
            found = spans(probabilities[i], offsets[i], valid[i].cpu(), threshold)
            expected = {(e["start"], e["end"]) for e in row["result_list"]}
            tp += len(found & expected)
            predicted += len(found)
            gold += len(expected)
            if len(examples) < 5 and expected:
                examples.append({"prompt": row["prompt"], "content": row["content"], "gold": [row["content"][a:b] for a, b in sorted(expected)], "prediction": [row["content"][a:b] for a, b in sorted(found)]})
    p, r = tp / max(predicted, 1), tp / max(gold, 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9), "gold_spans": gold}, examples


def augment(rows, mode, seed, uie_base=False, filling_checkpoint="t5_filling"):
    rng = random.Random(seed)
    extra = []
    if mode == "none":
        return rows
    positive = [r for r in rows if r["result_list"]]
    if mode == "negative":
        # Mine false positives among already annotated negative TRAIN examples only.
        pretrained, tok = prepare_span(uie_base)
        baseline = SpanModel(pretrained).to(device())
        ckpt = torch.load(BASE / "local_runs/checkpoints" / ("uie_base_none" if uie_base else "uie_none") / "model.pt", map_location="cpu", weights_only=True)
        baseline.load_state_dict(ckpt["state"])
        baseline.eval()
        negatives = [r for r in rows if not r["result_list"]]
        with torch.no_grad():
            for batch in DataLoader(negatives, batch_size=16, collate_fn=list):
                inputs, offsets, valid, _, _ = encode(tok, batch, device())
                probabilities = baseline(inputs).sigmoid().cpu()
                for i, row in enumerate(batch):
                    if spans(probabilities[i], offsets[i], valid[i].cpu(), ckpt["threshold"]):
                        extra.append(row)
        del baseline
    elif mode == "swap":
        if not uie_base:
            # Preserve the earlier small-encoder control experiment.
            groups = {}
            for row in positive:
                if len(row["result_list"]) == 1:
                    groups.setdefault(row["prompt"], []).append(row)
            for prompt, items in groups.items():
                for row in items:
                    entity = row["result_list"][0]
                    replacement = rng.choice(items)["result_list"][0]["text"]
                    content = row["content"][:entity["start"]] + replacement + row["content"][entity["end"]:]
                    extra.append({"prompt": prompt, "content": content, "result_list": [{"text": replacement, "start": entity["start"], "end": entity["start"] + len(replacement)}]})
            return rows + extra
        groups = {}
        for row in positive:
            if len(row["result_list"]) == 1:
                if "的" in row["prompt"]:
                    subject, predicate = row["prompt"].rsplit("的", 1)
                    if subject and subject in row["content"]:
                        groups.setdefault(predicate, []).append(row)
        for predicate, items in groups.items():
            for row in items:
                other = [item for item in items if item["content"] != row["content"]]
                if not other:
                    continue
                donor = rng.choice(other)
                subject = row["prompt"].rsplit("的", 1)[0]
                donor_subject = donor["prompt"].rsplit("的", 1)[0]
                obj, donor_obj = row["result_list"][0], donor["result_list"][0]
                subject_begin = donor["content"].find(donor_subject)
                subject_end = subject_begin + len(donor_subject)
                if not (subject_end <= donor_obj["start"] or donor_obj["end"] <= subject_begin):
                    continue
                edits = [(subject_begin, subject_end, subject, False),
                         (donor_obj["start"], donor_obj["end"], obj["text"], True)]
                cursor, pieces, object_begin = 0, [], None
                for begin, end, replacement, is_object in sorted(edits):
                    pieces.append(donor["content"][cursor:begin])
                    if is_object:
                        object_begin = sum(len(piece) for piece in pieces)
                    pieces.append(replacement)
                    cursor = end
                pieces.append(donor["content"][cursor:])
                content = "".join(pieces)
                entity = {"text": obj["text"], "start": object_begin, "end": object_begin + len(obj["text"])}
                assert content[entity["start"]:entity["end"]] == entity["text"]
                extra.append({"prompt": subject + "的" + predicate, "content": content, "result_list": [entity]})
        out = ROOT / "运行记录/quality/swap_spo_samples.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"strategy": "same predicate; subject and object jointly inserted into donor context", "accepted_count": len(extra), "samples": extra}, ensure_ascii=False, indent=2), encoding="utf-8")
        print("SwapSPO", len(extra), "samples", flush=True)
    elif mode == "fill":
        extra = mask_fill(rows, seed, filling_checkpoint, uie_base)
    return rows + extra


def prepare_span(uie_base=False):
    if not uie_base:
        return prepare_model()
    folder = ROOT / ".cache/models/uie-base-zh"
    tok = BertTokenizerFast.from_pretrained(folder, local_files_only=True)
    return folder / "converted", tok


def mask_fill(rows, seed, filling_checkpoint="t5_filling", uie_base=False):
    rng = random.Random(seed)
    folder = BASE / "local_runs/checkpoints" / filling_checkpoint
    tok = BertTokenizerFast.from_pretrained(folder, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(folder, local_files_only=True).eval()
    candidates = [r for r in rows if r["result_list"] and len(r["content"]) > 12]
    annotated_spans = {}
    for row in rows:
        annotated_spans.setdefault(row["content"], set()).update((e["start"], e["end"]) for e in row["result_list"])
    rng.shuffle(candidates)
    generated, proposals = [], []
    for row in candidates[:150]:
        content = row["content"]
        protected = {i for begin, end in annotated_spans[content] for i in range(begin, end)}
        subject = row["prompt"].rsplit("的", 1)[0]
        if "的" in row["prompt"] and subject in content:
            position = content.index(subject)
            protected.update(range(position, position + len(subject)))
        starts = [i for i in range(len(content) - 3) if not protected.intersection(range(i, i + 3))]
        if not starts:
            continue
        begin = rng.choice(starts)
        end = begin + 3
        masked = content[max(0, begin - 60):begin] + "[MASK]" + content[end:end + 60]
        source = '"' + masked + '"中[MASK]位置的文本是：'
        inputs = tok(source, return_tensors="pt", truncation=True, max_length=192)
        inputs.pop("token_type_ids", None)
        assert int(inputs["input_ids"].eq(tok.mask_token_id).sum()) == 2
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=16, num_beams=2, no_repeat_ngram_size=3)
        replacement = tok.decode(ids[0], skip_special_tokens=True).replace(" ", "").strip()
        record = {"prompt": row["prompt"], "original_content": content, "input": source, "removed": content[begin:end], "generated": replacement}
        repeated = any(replacement.count(replacement[j:j + 3]) > 1 for j in range(max(0, len(replacement) - 2)))
        if not replacement or len(replacement) > 12 or replacement == content[begin:end] or repeated:
            generated.append(record | {"accepted": False, "reason": "empty, excessive length, repeated phrase, or unchanged filling"})
            continue
        text = content[:begin] + replacement + content[end:]
        entities = []
        for entity in row["result_list"]:
            delta = len(replacement) - (end - begin) if entity["start"] >= end else 0
            value = entity | {"start": entity["start"] + delta, "end": entity["end"] + delta}
            assert text[value["start"]:value["end"]] == value["text"]
            entities.append(value)
        augmented = row | {"content": text, "result_list": entities}
        proposals.append((row, augmented, len(generated)))
        generated.append(record | {"accepted": True, "content": text, "result_list": entities})
    del model
    accepted = []
    if proposals:
        pretrained, span_tok = prepare_span(uie_base)
        reference = SpanModel(pretrained).to(device()).eval()
        path = BASE / "local_runs/checkpoints" / ("uie_base_none" if uie_base else "uie_none") / "model.pt"
        snapshot = torch.load(path, map_location="cpu", weights_only=True)
        reference.load_state_dict(snapshot["state"])
        threshold = snapshot["threshold"]
        for old, new, index in proposals:
            inputs, offsets, valid, _, _ = encode(span_tok, [old, new], device())
            with torch.no_grad():
                probabilities = reference(inputs).sigmoid().cpu()
            predictions = []
            for i, item in enumerate([old, new]):
                found = spans(probabilities[i], offsets[i], valid[i].cpu(), threshold)
                predictions.append({item["content"][a:b] for a, b in found})
            consistent = predictions[0] == predictions[1]
            generated[index]["reference_consistent"] = consistent
            generated[index]["reference_predictions"] = [sorted(values) for values in predictions]
            generated[index]["accepted"] = consistent
            if consistent:
                accepted.append(new)
            else:
                generated[index]["reason"] = "baseline extraction changed after filling"
        del reference, snapshot
    out = ROOT / "运行记录/quality/mask_fill_samples.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": filling_checkpoint, "rules": "60-character context on each side; both MASK tokens retained; length<=12; no repeated trigrams; original entity offsets preserved; baseline extraction text set unchanged", "attempts": generated, "accepted_count": len(accepted)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Mask Then Fill", len(generated), "generated", len(accepted), "accepted", flush=True)
    return accepted


def main(args):
    seed_all(args.seed)
    pretrained, tok = prepare_span(args.pretrained_uie)
    rows = data(args.seed)
    original_count = len(rows["train"])
    rows["train"] = augment(rows["train"], args.augment, args.seed, args.pretrained_uie, args.filling_checkpoint)
    # Mining instantiates a baseline and consumes RNG. Reset before the new model
    # so all augmentation arms start from identical randomly initialized heads.
    seed_all(args.seed)
    dev = device()
    model = SpanModel(pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    run_name = f"uie_base_{args.augment}_v1" if args.pretrained_uie else f"uie_{args.augment}_v3"
    run = Run(BASE / "local_runs/results", run_name, vars(args) | {"pretrained": "Pky/uie-base-zh pretrained encoder and boundary heads" if args.pretrained_uie else "uer/chinese_roberta_L-2_H-128", "counts": {s: len(r) for s, r in rows.items()}, "original_train_count": original_count, "initialization": "seed reset after mining, before model construction; same initial heads across arms", "negative_mining": "baseline false positives on already labeled training negatives only", "split": "group by original content; upstream empty test replaced by held-out half of dev"})
    baseline_candidates = [(evaluate(model, tok, rows["dev"], dev, args.batch, t)[0], t) for t in [.3, .5, .7]]
    baseline, baseline_threshold = max(baseline_candidates, key=lambda pair: pair[0]["f1"])
    best, state, threshold, best_epoch = -1., None, .5, None
    if args.pretrained_uie:
        best, threshold, best_epoch = baseline["f1"], baseline_threshold, 0
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for epoch in range(args.epochs):
        model.train()
        total, count = 0., 0
        for batch in DataLoader(rows["train"], batch_size=args.batch, shuffle=True, collate_fn=list):
            inputs, offsets, valid, start, end = encode(tok, batch, dev)
            opt.zero_grad()
            logits = model(inputs)
            target = torch.stack([start, end], -1)
            # Weight sparse positive boundaries; mask every prompt/special/padding token.
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=torch.tensor(10., device=dev))
            loss = (loss * valid.unsqueeze(-1)).sum() / (2 * valid.sum().clamp_min(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            total += float(loss.detach()) * len(batch)
            count += len(batch)
        candidates = [(evaluate(model, tok, rows["dev"], dev, args.batch, t)[0], t) for t in [.3, .5, .7]]
        metrics, selected = max(candidates, key=lambda pair: pair[0]["f1"])
        run.epoch(epoch=epoch + 1, loss=total / count, dev=metrics, threshold=selected)
        if metrics["f1"] > best:
            best, state, threshold, best_epoch = metrics["f1"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, selected, epoch + 1
    model.load_state_dict(state)
    metrics, examples = evaluate(model, tok, rows["test"], dev, args.batch, threshold)
    ckpt = BASE / "local_runs/checkpoints" / (("uie_base_" if args.pretrained_uie else "uie_") + args.augment)
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.cpu().state_dict(), "config": vars(args), "threshold": threshold}, ckpt / "model.pt")
    run.finish(baseline_dev=baseline, best_dev_f1=best, best_epoch=best_epoch, test=metrics, examples=examples, threshold=threshold)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--augment", choices=["none", "negative", "swap", "fill"], default="none")
    p.add_argument("--pretrained-uie", action="store_true")
    p.add_argument("--filling-checkpoint", default="t5_filling")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
