"""Audit original task coverage, final experiment scales and report evidence."""
import hashlib
import itertools
import json
import re
from pathlib import Path
from urllib.parse import unquote
from submission_reports import ROOT, NLP, GRAPH, select
from plm_submission_reports import select_plm


def main():
    checks = []
    def record(name, passed, **details):
        checks.append({"name": name, "passed": bool(passed), **details})
        print(name, "PASS" if passed else "PENDING", flush=True)
    for task, expected in [("task1", 9), ("task2", 7)]:
        values = select(ROOT / "nlp-beginner/legacy" / task, "nlp", task)
        count = sum(d["config"].get("data_name") == "sentiment_kaggle" for _, d in values)
        record("NLP " + task + " original Kaggle data", count == expected, count=count, expected=expected)
    for task in ["task3", "task4"]:
        values = select(ROOT / "nlp-beginner/legacy" / task, "nlp", task)
        record("NLP " + task + " expanded experiment", any(d["name"].endswith("_expanded") for _, d in values))
        if task == "task3" and values:
            c = values[0][1]
            record("SNLI full held-out evaluation", c.get("counts", {}).get("train") == 100000 and c.get("counts", {}).get("test", 0) > 9000)
        if task == "task4" and values:
            record("CoNLL complete sentence coverage", all(d["config"]["max_length"] >= 124 for _, d in values))
            record("NER casing study", any(d["config"].get("use_case") for _, d in values))
    for task in ["task1", "task2"]:
        values = select(ROOT / "graph_beginner" / task, "graph", task)
        primary = [d for _, d in values if d["name"].endswith("_isolated")]
        grid = {(d["config"]["dataset"], d["config"]["model"], d["config"]["mode"]) for d in primary}
        wanted = set(itertools.product(["Cora", "CiteSeer", "Flickr"], ["gcn", "gat", "sage", "gin"], ["full", "sampled"]))
        record("Graph " + task + " model/dataset/mode grid", grid == wanted, count=len(primary))
        record("Graph " + task + " learning-rate/layer studies", sum(d["name"].endswith("_ablation") for _, d in values) == 4)
    values = select(ROOT / "graph_beginner/task3", "graph", "task3")
    primary = [d for _, d in values if not d["name"].endswith("_ablation")]
    grid = {(d["config"]["dataset"], d["config"]["model"], d["config"]["pool"], d["config"]["mode"]) for d in primary}
    record("Graph graph-level full pooling grid", grid == set(itertools.product(["MUTAG", "ZINC"], ["gcn", "gat", "sage", "gin"], ["mean", "max", "min"], ["full", "batch"])))
    record("Graph graph-level parameter studies", sum(d["name"].endswith("_ablation") for _, d in values) == 8)
    values = select(ROOT / "graph_beginner/task4", "graph", "task4")
    record("KGE full training and test", len(values) == 3 and all(d["config"].get("limit") == 0 and d.get("test", {}).get("evaluated_triples", 0) > 20000 for _, d in values))
    plm = select_plm()
    names = {d["name"] for _, d in plm}
    for prefix in ["classification", "pointwise", "dssm", "sbert", "simcse", "pet", "ptuning", "reward", "qwen_ppo", "qwen_mbti", "sft", "cpt", "scratch", "scratch-sft"]:
        record("PLM " + prefix, any(name.startswith(prefix) for name in names))
    record("UIE pretrained and augmentation grid", all("uie_base_" + arm + "_v1" in names for arm in ["none", "negative", "swap", "fill"]))
    record("T5 expanded QA and filling", all("t5_" + task + "_n5000_v2_expanded" in names for task in ["qa", "filling"]))
    record("Zero-shot expanded evaluation", "qwen_zero_shot_v2_expanded" in names)
    record("Zero-shot prompt validation study", any(d["name"] == "qwen_zero_shot_prompt_study" and len(d.get("candidates", [])) == 3 and d["config"]["counts"]["test"] == 100 for _, d in plm))
    zero_checks = ROOT / "运行记录/诊断/zero_shot_output_checks.json"
    record("Zero-shot source-text output check", zero_checks.exists() and json.loads(zero_checks.read_text(encoding="utf-8"))["status"] == "passed")
    record("PPO held-out before/after evaluation", any(d["name"] == "qwen_ppo_v2_expanded" and d.get("test", {}).get("samples") == 16 and d.get("baseline_test", {}).get("samples") == 16 for _, d in plm))
    fill = ROOT / "运行记录/quality/mask_fill_samples.json"
    record("Actual Mask Then Fill generation", fill.exists() and json.loads(fill.read_text(encoding="utf-8")).get("accepted_count", 0) > 0)
    if fill.exists():
        generated = json.loads(fill.read_text(encoding="utf-8"))
        accepted = [row for row in generated["attempts"] if row["accepted"]]
        record("Mask filling quality and offsets", generated["model"] == "t5_filling_expanded" and len(accepted) == generated["accepted_count"] and all(row.get("reference_consistent") and 0 < len(row["generated"]) <= 12 and all(row["content"][e["start"]:e["end"]] == e["text"] for e in row["result_list"]) for row in accepted))
    swap = ROOT / "运行记录/quality/swap_spo_samples.json"
    record("Actual SwapSPO augmentation", swap.exists() and json.loads(swap.read_text(encoding="utf-8")).get("accepted_count", 0) > 0)
    for name in ["task2_classes.json", "task3_classes.json", "task4_entities.json"]:
        path = ROOT / "运行记录/诊断" / name
        record("Held-out diagnostics " + name, path.exists() and json.loads(path.read_text(encoding="utf-8")).get("test_examples", 0) > 1000)
    attention = ROOT / "运行记录/诊断/esim_attention.json"
    record("ESIM attention on held-out example", attention.exists() and all(abs(value - 1) < 1e-5 for value in json.loads(attention.read_text(encoding="utf-8"))["row_sums"]))
    link_reload = ROOT / "运行记录/诊断/link_reload_devices.json"
    record("Link saved-model CPU/CUDA comparison", link_reload.exists() and json.loads(link_reload.read_text(encoding="utf-8"))["status"] == "passed")
    source = ROOT / "nlp-beginner/legacy/data/processed/sentiment_kaggle"
    keys = {s: {json.loads(line)["sentence_id"] for line in (source / (s + ".jsonl")).read_text(encoding="utf-8").splitlines()} for s in ["train", "dev", "test"]}
    record("Kaggle group split has no overlap", all(not keys[a] & keys[b] for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]))
    reports = [ROOT / "学习报告/总报告.md", ROOT / "学习报告/PLM任务.md"]
    reports += [ROOT / "nlp-beginner/legacy" / row[1] / "README.md" for row in NLP]
    reports += [ROOT / "graph_beginner" / row[1] / "README.md" for row in GRAPH]
    reports += list((ROOT / "学习报告").glob("PLM_*.md"))
    broken = []
    for report in reports:
        text = report.read_text(encoding="utf-8")
        if report.name != "PLM任务.md":
            record("Report sections " + report.relative_to(ROOT).as_posix(), all(term in text for term in ["实验环境与条件", "实验方法与过程", "实验结果及分析", "遇到的问题及处理情况"]))
        record("Portable links " + report.relative_to(ROOT).as_posix(), "C:/Users/" not in text and "C:\\Users\\" not in text)
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if target.startswith(("http:", "https:", "#")):
                continue
            path = report.parent / unquote(target.split("#")[0])
            if not path.exists():
                broken.append({"report": str(report.relative_to(ROOT)), "link": target})
    record("Report links exist", not broken, broken=broken)
    screenshots = ["nlp" + str(i) for i in range(1, 6)] + ["graph" + str(i) for i in range(1, 5)] + ["plm-bert", "plm-uie", "plm-reward", "plm-qa", "plm-filling", "plm-sft", "plm-cpt", "plm-ppo", "plm-scratch", "plm-scratch-sft", "plm-zero-shot", "plm-mbti", "plm-tools"]
    for name in screenshots:
        path = ROOT / "学习报告/运行截图" / name
        present = all(path.with_suffix(ext).exists() for ext in [".png", ".txt", ".json"])
        valid = False
        if present:
            record_data = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            original = path.with_suffix(".txt").read_text(encoding="utf-8")
            valid = record_data["exit_code"] == 0 and hashlib.sha256(original.encode()).hexdigest() == record_data["output_sha256"]
            if record_data.get("screenshot_sha256"):
                valid = valid and hashlib.sha256(path.with_suffix(".png").read_bytes()).hexdigest() == record_data["screenshot_sha256"]
        record("Raw screenshot " + name, present and valid)
    for name in ["运行记录/核查/audit.json", "运行记录/核查/generation_reload_checks.json", "运行记录/local_tools_checks.json", "运行记录/核查/kge_ranking.json", "运行记录/独立环境核查.json"]:
        path = ROOT / name
        record(name, path.exists() and json.loads(path.read_text(encoding="utf-8"))["status"] == "passed")
    source_manifest = ROOT / "运行记录/数据模型来源.json"
    record("Data and model provenance manifest", source_manifest.exists() and len(json.loads(source_manifest.read_text(encoding="utf-8"))["files"]) >= 40)
    result = {"status": "passed" if all(c["passed"] for c in checks) else "pending", "checks": checks,
              "scope": "all original task categories with documented local model/data conditions; human review remains separate"}
    (ROOT / "运行记录/提交核查.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result["status"], flush=True)
    return result["status"] == "passed"


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
