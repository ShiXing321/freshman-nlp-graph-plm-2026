"""Evaluate the source-text constraint on saved zero-shot extraction outputs."""
import json
from pathlib import Path
from zero_shot_prompt_study import metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports = []
    for name in ["qwen_zero_shot_v2_expanded", "qwen_zero_shot_prompt_study"]:
        result = json.loads((ROOT / "transformers_tasks/local_runs/results" / (name + ".json")).read_text(encoding="utf-8"))
        assert result["status"] == "completed"
        examples = result["tasks"]["extraction"]["examples"]
        processed, records, removed, invalid_gold = [], [], 0, []
        for row in examples:
            if any(entity not in row["content"] for entity in row["gold"]):
                invalid_gold.append(row["prompt"])
            predicted = row["prediction"]
            found = None if predicted is None else sorted({entity for entity in predicted if entity in row["content"]})
            removed += len(set(predicted or [])) - len(found or [])
            processed.append(row | {"prediction": found, "correct": found is not None and found == sorted(set(row["gold"]))})
            records.append(row | {"source_checked_prediction": found})
        score = metrics(processed)
        raw = metrics(examples)
        assert invalid_gold or abs(raw["recall"] - score["recall"]) < 1e-12
        reports.append({"model": name, "raw": raw, "source_checked": score,
            "removed_entities": removed, "invalid_gold_mentions": invalid_gold, "examples": records})
    status = "passed" if all(not r["invalid_gold_mentions"] for r in reports) else "failed"
    output = {"status": status, "method": "Keep a parsed entity only when its exact text occurs in the original content; invalid JSON stays invalid. No new model generation or label-based filtering.", "reports": reports}
    (ROOT / "运行记录/诊断/zero_shot_output_checks.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "models": [{"name": r["model"], "removed_entities": r["removed_entities"], "raw_f1": r["raw"]["f1"], "checked_f1": r["source_checked"]["f1"]} for r in reports]}, ensure_ascii=False))
    return status == "passed"


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
