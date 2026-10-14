"""Compare relation-extraction prompts on dev and retain the fixed test sample."""
import argparse
import ctypes
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "transformers_tasks/local_runs")]


def prompt(row, template):
    subject, relation = row["prompt"].rsplit("的", 1) if "的" in row["prompt"] else ("原文中的实体", row["prompt"])
    if template == 0:
        return f"抽取原文中属于“{row['prompt']}”的信息。只输出JSON字符串数组，没有则输出[]。原文：{row['content']}"
    if template == 1:
        return f"任务：关系抽取。\n主体：{subject}\n关系：{relation}\n原文：{row['content']}\n只抽取原文明确说明的该主体在该关系下的客体。无对应事实返回[]。请只输出JSON字符串数组。"
    return f"核对关系：主体“{subject}”的“{relation}”。\n原文：{row['content']}\n若原文没有明确给出这项属性，输出[]。原文出现的其他人物、地点、时间只在符合当前关系时返回。请逐字摘录客体，只输出JSON字符串数组。"


def metrics(examples):
    tp = predicted = gold = 0
    positive = [r for r in examples if r["gold"]]
    negative = [r for r in examples if not r["gold"]]
    for row in examples:
        expected = set(row["gold"])
        found = set(row["prediction"] or [])
        tp += len(expected & found)
        predicted += len(found)
        gold += len(expected)
    precision, recall = tp / max(predicted, 1), tp / max(gold, 1)
    pacc = sum(r["correct"] for r in positive) / max(len(positive), 1)
    nacc = sum(r["correct"] for r in negative) / max(len(negative), 1)
    return {"exact_set_accuracy": sum(r["correct"] for r in examples) / len(examples),
        "valid_json_rate": sum(r["prediction"] is not None for r in examples) / len(examples),
        "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "positive_samples": len(positive), "negative_samples": len(negative),
        "positive_exact_set_accuracy": pacc, "negative_exact_set_accuracy": nacc,
        "balanced_exact_set_accuracy": (pacc + nacc) / 2,
        "empty_output_rate": sum(r["prediction"] == [] for r in examples) / len(examples)}


def main(args):
    if args.wait_pid:
        handle = ctypes.windll.kernel32.OpenProcess(0x00100000, False, args.wait_pid)
        if handle:
            ctypes.windll.kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
            ctypes.windll.kernel32.CloseHandle(handle)
    from common.experiment import Run, seed_all
    from llm_tasks import qwen, chat
    from span_task import data
    seed_all(args.seed)
    rows = data(args.seed)
    rng = random.Random(args.seed)
    balanced = []
    for positive in [True, False]:
        group = [row for row in rows["dev"] if bool(row["result_list"]) == positive]
        balanced.extend(rng.sample(group, min(20, len(group))))
    test = json.loads((ROOT / "transformers_tasks/local_runs/results/qwen_zero_shot_v2_expanded.json").read_text(encoding="utf-8"))["tasks"]["extraction"]["examples"]
    assert not {r["content"] for r in balanced} & {r["content"] for r in test}
    model, tok = qwen()
    run = Run(ROOT / "transformers_tasks/local_runs/results", "qwen_zero_shot_prompt_study", {"seed": args.seed,
        "task": "zero-shot prompt study", "model": "Qwen2.5-0.5B-Instruct", "few_shot_examples": 0,
        "counts": {"train": 0, "dev": len(balanced), "test": len(test)},
        "selection": "dev entity-text F1, then balanced exact-set accuracy; fixed test rows from expanded baseline",
        "maximum_new_tokens": 80})
    def evaluate(values, template):
        examples = []
        for row in values:
            answer = chat(model, tok, prompt(row, template), 80)
            match = re.search(r"\[.*?\]", answer, re.S)
            try:
                found = json.loads(match.group()) if match else None
                if not isinstance(found, list) or any(not isinstance(t, str) for t in found):
                    found = None
            except (ValueError, AttributeError):
                found = None
            gold = sorted(set(row.get("gold", [e["text"] for e in row.get("result_list", [])])))
            examples.append({"prompt": row["prompt"], "content": row["content"], "gold": gold, "output": answer,
                "prediction": found, "correct": found is not None and sorted(set(found)) == gold})
        return metrics(examples), examples
    candidates = []
    for template in range(3):
        score, examples = evaluate(balanced, template)
        candidates.append({"template": template, "metrics": score, "examples": examples})
        run.epoch(template=template, dev=score)
    chosen = max(candidates, key=lambda row: (row["metrics"]["f1"], row["metrics"]["balanced_exact_set_accuracy"], -row["template"]))
    score, examples = evaluate(test, chosen["template"])
    baseline = metrics(test)
    run.finish(selected_template=chosen["template"], candidates=candidates, best_dev_f1=chosen["metrics"]["f1"],
        test=score, baseline_test=baseline, tasks={"extraction": score | {"examples": examples}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
