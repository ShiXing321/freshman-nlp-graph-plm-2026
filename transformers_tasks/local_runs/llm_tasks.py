"""Local small-model versions of zero-shot, MBTI, LoRA, CPT and scratch training."""
import argparse
import copy
import hashlib
import itertools
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
sys.path.insert(0, str(ROOT))
from common.experiment import Run, device, seed_all


def json_rows(path, limit=0):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(l) for l in itertools.islice((l for l in f if l.strip()), limit or None)]


def pretrain_rows(limit, seed):
    folder = BASE / "LLM/LLMsTrainer/data/shuffled_data/pretrain"
    records = []
    for path in sorted(folder.glob("*.jsonl")):
        records.extend({"text": r["text"]} for r in json_rows(path, max(limit * 2, 500)))
    records = list({r["text"]: r for r in records if r["text"].strip()}.values())
    random.Random(seed).shuffle(records)
    n = len(records)
    splits = {"train": records[:int(n * .8)][:limit], "dev": records[int(n * .8):int(n * .9)][:50], "test": records[int(n * .9):][:50]}
    return splits


def sft_rows(limit, seed, scratch=False):
    if scratch:
        path = BASE / "LLM/LLMsTrainer/data/shuffled_data/sft/sharegpt.jsonl"
        records = []
        for r in json_rows(path, limit * 4):
            dialog = r.get("dialog", [])
            for a, b in zip(dialog, dialog[1:]):
                if a["role"] == "user" and b["role"] == "assistant":
                    records.append({"input": a["content"], "target": b["content"]})
                    break
        records = list({r["input"]: r for r in records}.values())
        random.Random(seed).shuffle(records)
        n = len(records)
        return {"train": records[:int(n * .8)][:limit], "dev": records[int(n * .8):int(n * .9)][:30], "test": records[int(n * .9):][:30]}
    folder = BASE / "LLM/chatglm_finetune/data"
    train = [{"input": r["context"], "target": r["target"]} for r in json_rows(folder / "mixed_train_dataset.jsonl")]
    val = [{"input": r["context"], "target": r["target"]} for r in json_rows(folder / "mixed_dev_dataset.jsonl")]
    val = list({r["input"]: r for r in val}.values())
    random.Random(seed).shuffle(val)
    holdout = {r["input"] for r in val}
    train = list({r["input"]: r for r in train if r["input"] not in holdout}.values())
    rng = random.Random(seed)
    return {"train": rng.sample(train, min(len(train), limit)), "dev": val[:len(val)//2][:30], "test": val[len(val)//2:][:30]}


def scratch_tokenizer(rows, folder):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=3000, min_frequency=2, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"])
    tokenizer.train_from_iterator([r["text"] for r in rows], trainer)
    tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>", unk_token="<unk>")
    tok.save_pretrained(folder)
    return tok


def qwen():
    folder = ROOT / ".cache/models/qwen05"
    tok = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(folder, local_files_only=True, dtype=torch.bfloat16 if device().type == "cuda" else torch.float32, attn_implementation="sdpa").to(device())
    return model, tok


@torch.no_grad()
def chat(model, tok, prompt, maximum=48):
    model.eval()
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) if tok.chat_template else "用户：" + prompt + "\n助手："
    capacity = getattr(model.config, "max_position_embeddings", getattr(model.config, "n_positions", 1024))
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=min(384, capacity - maximum - 1)).to(next(model.parameters()).device)
    output = model.generate(**inputs, max_new_tokens=maximum, do_sample=False, use_cache=True, pad_token_id=tok.pad_token_id)
    return tok.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def zero_shot(args):
    from bert_tasks import load_rows
    from span_task import data
    model, tok = qwen()
    mapping = dict(line.split(",", 1) for line in (BASE / "text_classification/data/comment_classify/label_mapping.txt").read_text(encoding="utf-8").splitlines() if line)
    run = Run(BASE / "local_runs/results", "qwen_zero_shot_v2" + args.tag, vars(args) | {"model": "Qwen2.5-0.5B-Instruct; replaces upstream ChatGLM-6B", "few_shot_examples": 0, "category_schema": mapping})
    tasks = {}
    rows, labels = load_rows("classification", args.seed)
    labels = [mapping[label] for label in labels]
    rng = random.Random(args.seed)
    examples = []
    for r in rng.sample(rows["test"], min(args.eval_size, len(rows["test"]))):
        prompt = "将评论分类。只回答一个类别。类别：" + "、".join(labels) + "。评论：" + r["text"]
        answer = chat(model, tok, prompt, 12)
        examples.append({"input": r["text"], "gold": labels[r["label"]], "output": answer, "correct": answer.strip(" 。\n") == labels[r["label"]]})
    tasks["classification"] = {"accuracy": sum(e["correct"] for e in examples) / len(examples), "examples": examples}
    rows, _ = load_rows("simcse", args.seed)
    examples = []
    for r in rng.sample(rows["test"], min(args.eval_size, len(rows["test"]))):
        prompt = f"判断两个句子表达的意思是否相同，只回答相同或不同。句子1：{r['a']}。句子2：{r['b']}"
        answer = chat(model, tok, prompt, 12)
        predicted = 0 if "不同" in answer else 1 if "相同" in answer else -1
        examples.append({"a": r["a"], "b": r["b"], "gold": r["label"], "output": answer, "prediction": predicted, "correct": predicted == r["label"]})
    tasks["matching"] = {"accuracy": sum(e["correct"] for e in examples) / len(examples), "examples": examples}
    examples = []
    for r in rng.sample(data(args.seed)["test"], min(args.eval_size, len(data(args.seed)["test"]))):
        prompt = f"抽取原文中属于“{r['prompt']}”的信息。只输出JSON字符串数组，没有则输出[]。原文：{r['content']}"
        answer = chat(model, tok, prompt, 80)
        match = re.search(r"\[.*?\]", answer, re.S)
        try:
            predicted = json.loads(match.group()) if match else None
            if not isinstance(predicted, list) or any(not isinstance(t, str) for t in predicted):
                predicted = None
        except (ValueError, AttributeError):
            predicted = None
        gold = sorted({e["text"] for e in r["result_list"]})
        examples.append({"prompt": r["prompt"], "content": r["content"], "gold": gold, "output": answer, "prediction": predicted, "correct": predicted is not None and sorted(set(predicted)) == gold})
    tasks["extraction"] = {"exact_set_accuracy": sum(e["correct"] for e in examples) / len(examples), "valid_json_rate": sum(e["prediction"] is not None for e in examples) / len(examples), "examples": examples}
    run.finish(tasks=tasks)


def mbti(args):
    model, tok = qwen()
    questions = json.loads((BASE / "LLM/llms_mbti/mbti_questions.json").read_text(encoding="utf-8"))
    run = Run(BASE / "local_runs/results", "qwen_mbti", vars(args) | {"model": "Qwen2.5-0.5B-Instruct", "meaning": "prompt-conditioned model responses, not a psychological diagnosis"})
    examples, votes = [], Counter()
    for number, question in questions.items():
        answer = chat(model, tok, "请选择A或B，只输出一个字母。\n" + question["question"].replace("\\n", "\n"), 8)
        match = re.search(r"(?<![A-Za-z])[AB](?![A-Za-z])", answer)
        choice = match.group() if match else None
        if choice:
            votes[question[choice]] += 1
        examples.append({"number": number, "question": question["question"], "output": answer, "choice": choice})
        if len(examples) % 20 == 0:
            run.epoch(answered=len(examples), valid=sum(votes.values()))
    profile = "".join(a if votes[a] > votes[b] else b if votes[b] > votes[a] else "?" for a, b in [("E", "I"), ("S", "N"), ("T", "F"), ("J", "P")])
    run.finish(profile=profile, votes=dict(votes), valid_responses=sum(votes.values()), total_questions=len(questions), examples=examples)


def training(args):
    scratch = args.task in ["scratch", "scratch-sft"]
    instruction = args.task in ["sft", "scratch-sft"]
    rows = sft_rows(args.limit, args.seed, scratch) if instruction else pretrain_rows(args.limit, args.seed)
    ckpt = BASE / "local_runs/checkpoints" / args.task
    if args.task == "scratch":
        tok = scratch_tokenizer(rows["train"], ckpt)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tok), n_positions=args.length, n_embd=128, n_layer=4, n_head=4, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)).to(device())
    elif args.task == "scratch-sft":
        source = BASE / "local_runs/checkpoints/scratch"
        tok = AutoTokenizer.from_pretrained(source, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True).to(device())
    else:
        from peft import LoraConfig, get_peft_model
        model, tok = qwen()
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    dev = device()
    if instruction:
        def encode(row):
            text = tok.apply_chat_template([{"role": "user", "content": row["input"]}], tokenize=False, add_generation_prompt=True) if tok.chat_template else "用户：" + row["input"] + "\n助手："
            prefix = tok.encode(text, add_special_tokens=False)[-(args.length // 2):]
            target = tok.encode(row["target"], add_special_tokens=False)[:args.length - len(prefix) - 1] + [tok.eos_token_id]
            return prefix + target, [-100] * len(prefix) + target
    else:
        def encode(row):
            ids = tok.encode(row["text"], add_special_tokens=False)[:args.length - 1] + [tok.eos_token_id]
            return ids, ids.copy()
    encoded = {s: [encode(r) for r in values] for s, values in rows.items()}
    def collate(batch):
        n = max(len(x) for x, _ in batch)
        ids = torch.tensor([x + [tok.pad_token_id] * (n - len(x)) for x, _ in batch], device=dev)
        labels = torch.tensor([y + [-100] * (n - len(y)) for _, y in batch], device=dev)
        attention = torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x, _ in batch], device=dev)
        return {"input_ids": ids, "attention_mask": attention, "labels": labels}
    run = Run(BASE / "local_runs/results", args.task, vars(args) | {"model": "random GPT2 4-layer/128 hidden, custom byte BPE" if scratch else "Qwen2.5-0.5B-Instruct + LoRA", "parameters": sum(p.numel() for p in model.parameters()), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "counts": {s: len(r) for s, r in rows.items()}, "masking": "assistant target only; prompt and padding=-100" if instruction else "all nonpadding tokens", "split": "deduplicated document/prompt split, fixed seed, no test-based selection", "truncation": "prefix capped to half context; target truncated to remaining capacity"})
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    @torch.no_grad()
    def evaluate(split):
        model.eval()
        total, tokens = 0., 0
        for batch in DataLoader(encoded[split], batch_size=args.batch, collate_fn=collate):
            loss = model(**batch).loss
            count = int(batch["labels"][:, 1:].ne(-100).sum())
            total += float(loss) * count
            tokens += count
        loss = total / max(tokens, 1)
        return {"loss": loss, "perplexity": math.exp(min(loss, 80)), "tokens": tokens}
    baseline = evaluate("dev")
    best, state = baseline["loss"], {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
    examples_before = [{"input": r["input"], "gold": r["target"], "output": chat(model, tok, r["input"], 40)} for r in rows["test"][:3]] if instruction else []
    for epoch in range(args.epochs):
        model.train()
        total, count = 0., 0
        loader = DataLoader(encoded["train"], batch_size=args.batch, shuffle=True, collate_fn=collate)
        opt.zero_grad()
        for step, batch in enumerate(loader):
            loss = model(**batch).loss
            (loss / args.accumulate).backward()
            if (step + 1) % args.accumulate == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
                opt.zero_grad()
            total += float(loss.detach())
            count += 1
        metrics = evaluate("dev")
        run.epoch(epoch=epoch + 1, loss=total / count, dev=metrics)
        if metrics["loss"] < best:
            best = metrics["loss"]
            state = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
    with torch.no_grad():
        for k, p in model.named_parameters():
            if k in state:
                p.copy_(state[k].to(p.device))
    metrics = evaluate("test")
    examples = [{"input": r["input"], "gold": r["target"], "output": chat(model, tok, r["input"], 40)} for r in rows["test"][:5]] if instruction else [{"input": t, "output": chat(model, tok, t, 40)} for t in ["春天来了", "请解释梯度下降。"]]
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    run.finish(baseline_dev=baseline, best_dev_loss=best, test=metrics, examples_before=examples_before, examples=examples)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["zero-shot", "mbti", "sft", "cpt", "scratch", "scratch-sft"])
    p.add_argument("--limit", type=int, default=400)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--accumulate", type=int, default=8)
    p.add_argument("--length", type=int, default=192)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-size", type=int, default=20)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    seed_all(args.seed)
    if args.task == "zero-shot":
        zero_shot(args)
    elif args.task == "mbti":
        mbti(args)
    else:
        training(args)
