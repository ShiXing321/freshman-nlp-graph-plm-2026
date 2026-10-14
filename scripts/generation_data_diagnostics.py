"""Measure actual tokenizer length conditions used by the generation runs."""
import json
import sys
from pathlib import Path
import numpy as np
from transformers import BertTokenizerFast
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "transformers_tasks/local_runs")]
from seq2seq_tasks import load_data


def main():
    tok = BertTokenizerFast(vocab_file=str(ROOT / ".cache/models/t5-small-zh/vocab.txt"), do_lower_case=False)
    tok.model_max_length = 1000000
    reports = []
    paths = [ROOT / "transformers_tasks/local_runs/results" / ("t5_" + task + suffix + ".json")
        for task in ["qa", "filling"] for suffix in ["_n1000_v2", "_n5000_v2_expanded"]]
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        c = result["config"]
        rows = load_data(c["task"], c["seed"], c["limit"], c.get("eval_limit", 100))
        splits = {}
        for split, values in rows.items():
            source = tok([r["input"] for r in values], truncation=False)["input_ids"]
            target = tok([r["target"] for r in values], add_special_tokens=False, truncation=False)["input_ids"]
            lengths = [len(ids) for ids in source]
            splits[split] = {"examples": len(values), "source_token_mean": float(np.mean(lengths)),
                "source_token_p95": float(np.percentile(lengths, 95)), "source_token_max": max(lengths),
                "source_truncated": sum(n > c["source_length"] for n in lengths),
                "target_truncated": sum(len(ids) + 1 > c["target_length"] for ids in target)}
        reports.append({"name": result["name"], "source_length": c["source_length"], "target_length": c["target_length"], "splits": splits})
    baseline = ROOT / "运行记录/诊断/mask_fill_baseline_samples.json"
    mask = {}
    if baseline.exists():
        attempts = json.loads(baseline.read_text(encoding="utf-8"))["attempts"]
        inputs = tok([row["input"] for row in attempts], truncation=False)["input_ids"]
        mask = {"attempts": len(inputs), "source_truncated": sum(len(ids) > 192 for ids in inputs),
            "main_mask_missing": sum(tok.mask_token_id not in ids[:192] for ids in inputs),
            "instruction_or_main_mask_missing": sum(ids[:192].count(tok.mask_token_id) < 2 for ids in inputs)}
    output = ROOT / "运行记录/诊断/generation_lengths.json"
    output.write_text(json.dumps({"t5": reports, "baseline_mask_filling": mask}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"runs": len(reports), "baseline_mask_filling": mask}, ensure_ascii=False))


if __name__ == "__main__":
    main()
