"""Restore saved causal/T5 models for a direct, reproducible inference command."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration
from llm_tasks import qwen, chat, device, seed_all

BASE = Path(__file__).resolve().parents[1]

def inference(task, text, checkpoint=None):
    folder = Path(checkpoint) if checkpoint else BASE / "local_runs/checkpoints" / task
    if task in ["sft", "cpt", "ppo"]:
        from peft import PeftModel
        model, tok = qwen()
        model = PeftModel.from_pretrained(model, folder, local_files_only=True)
        output = chat(model, tok, text, 48)
    elif task in ["scratch", "scratch-sft"]:
        tok = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(folder, local_files_only=True).to(device())
        output = chat(model, tok, text, 40)
    else:
        tok = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        model = T5ForConditionalGeneration.from_pretrained(folder, local_files_only=True).to(device())
        model.eval()
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=192).to(device())
        inputs.pop("token_type_ids", None)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        output = tok.decode(ids[0], skip_special_tokens=True)
    return {"task": task, "input": text, "output": output, "checkpoint": str(folder)}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["sft", "cpt", "ppo", "scratch", "scratch-sft", "t5_qa", "t5_filling"])
    p.add_argument("--text", default="请用一句话解释梯度下降。")
    p.add_argument("--output", help="Optional JSON output file")
    p.add_argument("--checkpoint", help="Explicit checkpoint directory")
    args = p.parse_args()
    seed_all(42)
    result = inference(args.task, args.text, args.checkpoint)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
