"""Minimal token-level PPO: frozen reference, reward model, value head and GAE."""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn
from peft import LoraConfig, get_peft_model

from llm_tasks import qwen
from reward_task import RewardModel, prepare_model

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE.parent))
from common.experiment import Run, seed_all, device


def clipped_policy_loss(new, old, advantages, epsilon=.2):
    ratios = (new - old).exp()
    return -torch.minimum(ratios * advantages, ratios.clamp(1 - epsilon, 1 + epsilon) * advantages).mean()


def gae(rewards, values, gamma=1., lam=.95):
    advantage = torch.zeros_like(rewards)
    carry = torch.zeros((), device=rewards.device)
    for t in reversed(range(len(rewards))):
        future = values[t + 1] if t + 1 < len(values) else torch.zeros_like(carry)
        delta = rewards[t] + gamma * future - values[t]
        carry = delta + gamma * lam * carry
        advantage[t] = carry
    return advantage, advantage + values


def main(args):
    seed_all(args.seed)
    dev = device()
    base, tok = qwen()
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0., target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    value_head = nn.Linear(model.config.hidden_size, 1).to(dev)
    pretrained, reward_tok = prepare_model()
    reward_model = RewardModel(pretrained).to(dev)
    checkpoint = torch.load(BASE / "local_runs/checkpoints/reward/model.pt", map_location="cpu", weights_only=True)
    reward_model.load_state_dict(checkpoint["state"])
    reward_model.eval().requires_grad_(False)
    parameters = [p for p in model.parameters() if p.requires_grad] + list(value_head.parameters())
    opt = torch.optim.AdamW(parameters, lr=args.lr)
    run = Run(BASE / "local_runs/results", "qwen_ppo_v2" + args.tag, vars(args) | {"policy": "Qwen2.5-0.5B + LoRA", "reference": "same frozen base with adapters disabled", "sampling": "full categorical policy: temperature=1 top_p=1 top_k=0, no forced minimum length", "reward": "trained Chinese preference BERT checkpoint", "loss": "clipped token policy + clipped value; terminal learned reward - beta * sampled token KL", "algorithm": "GAE gamma=1 lambda=.95; whiten advantages over rollout tokens", "counts": {"train": args.steps * 4, "dev": 0, "test": 16 if args.heldout_eval else 0}})
    prompts = ["写一条正面的购物评论，只写评论。商品：苹果。", "写一条正面的购物评论，只写评论。商品：手机。", "写一条正面的购物评论，只写评论。商品：衣服。", "写一条正面的购物评论，只写评论。商品：酒店。"]
    all_samples = []
    def heldout():
        products = ["书籍", "水杯", "跑鞋", "键盘"]
        prompts = [template.format(item=item) for item in products for template in ["写一条正面的购物评论，只写评论。商品：{item}。", "请用一句话评价{item}的优点。"]]
        records = []
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if dev.type == "cuda" else None
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                for repeat in range(2):
                    torch.manual_seed(args.seed + 1000 + i * 2 + repeat)
                    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
                    inputs = tok(text, return_tensors="pt").to(dev)
                    sequence = model.generate(**inputs, max_new_tokens=args.response_length, do_sample=True, temperature=1., top_p=1., top_k=0, use_cache=True, pad_token_id=tok.pad_token_id)
                    response = tok.decode(sequence[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    reward = float(reward_model(reward_tok(response, return_tensors="pt", truncation=True, max_length=128).to(dev))[0])
                    records.append({"prompt": prompt, "repeat": repeat, "response": response, "reward": reward})
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        model.train(was_training)
        rewards = torch.tensor([row["reward"] for row in records])
        return {"reward_mean": float(rewards.mean()), "reward_std": float(rewards.std()), "samples": len(records)}, records
    baseline_test, baseline_examples = heldout() if args.heldout_eval else ({}, [])
    def forward(sequence, start, reference=False):
        output = model(input_ids=sequence, attention_mask=torch.ones_like(sequence), output_hidden_states=not reference, use_cache=False)
        logits = output.logits[0, start - 1:-1].float()
        chosen = sequence[0, start:]
        log_probs = logits.log_softmax(-1).gather(-1, chosen.unsqueeze(-1)).flatten()
        values = None if reference else value_head(output.hidden_states[-1][0, start - 1:-1].float()).flatten()
        return log_probs, values
    for iteration in range(args.steps):
        model.eval()
        rollout = []
        with torch.no_grad():
            for prompt in prompts:
                text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
                inputs = tok(text, return_tensors="pt").to(dev)
                sequence = model.generate(**inputs, max_new_tokens=args.response_length, do_sample=True, temperature=1., top_p=1., top_k=0, use_cache=True, pad_token_id=tok.pad_token_id)
                start = inputs["input_ids"].shape[1]
                old, values = forward(sequence, start)
                with model.disable_adapter():
                    reference, _ = forward(sequence, start, True)
                response = tok.decode(sequence[0, start:], skip_special_tokens=True)
                reward = float(reward_model(reward_tok(response, return_tensors="pt", truncation=True, max_length=128).to(dev))[0])
                rewards = -args.beta * (old - reference)
                rewards[-1] += max(-5., min(5., reward))
                advantage, returns = gae(rewards, values)
                rollout.append({"sequence": sequence, "start": start, "old": old.detach(), "values": values.detach(), "advantage": advantage.detach(), "returns": returns.detach()})
                all_samples.append({"iteration": iteration + 1, "prompt": prompt, "response": response, "reward": reward, "sampled_kl": float((old - reference).sum()), "tokens": len(old)})
        advantages = torch.cat([r["advantage"] for r in rollout])
        mean, std = advantages.mean(), advantages.std(unbiased=False).clamp_min(1e-6)
        for r in rollout:
            r["advantage"] = (r["advantage"] - mean) / std
        losses, clips = [], []
        model.train()
        for _ in range(args.ppo_epochs):
            random.shuffle(rollout)
            for r in rollout:
                new, values = forward(r["sequence"], r["start"])
                policy = clipped_policy_loss(new, r["old"], r["advantage"], args.clip)
                clipped_values = r["values"] + (values - r["values"]).clamp(-args.clip, args.clip)
                value_loss = .5 * torch.maximum((values - r["returns"]).square(), (clipped_values - r["returns"]).square()).mean()
                loss = policy + .5 * value_loss
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                opt.step()
                losses.append(float(loss.detach()))
                clips.append(float(((new.detach() - r["old"]).exp() - 1).abs().gt(args.clip).float().mean()))
        current = all_samples[-len(prompts):]
        run.epoch(iteration=iteration + 1, loss=sum(losses)/len(losses), clip_fraction=sum(clips)/len(clips), reward=sum(s["reward"] for s in current)/len(current), sampled_kl=sum(s["sampled_kl"] for s in current)/len(current))
    test, heldout_examples = heldout() if args.heldout_eval else ({}, [])
    ckpt = BASE / "local_runs/checkpoints" / ("ppo" + args.tag)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    torch.save(value_head.cpu().state_dict(), ckpt / "value_head.pt")
    run.finish(samples=all_samples, baseline_test=baseline_test, test=test, baseline_examples=baseline_examples, heldout_examples=heldout_examples)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--response-length", type=int, default=32)
    p.add_argument("--beta", type=float, default=.05)
    p.add_argument("--clip", type=float, default=.2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="")
    p.add_argument("--heldout-eval", action="store_true")
    main(p.parse_args())
