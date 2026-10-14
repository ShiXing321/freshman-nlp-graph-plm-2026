"""Validate the default dataset preparer without overwriting experiment inputs."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "nlp-beginner/legacy/prepare_data.py"
spec = importlib.util.spec_from_file_location("sentiment_prepare_check", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original = module.BASE
module.BASE = ROOT / "运行记录/数据来源核查"
module.sentiment()
checks = []
for split in ["train", "dev", "test"]:
    def load(base):
        return [(r["text"], r["label"]) for r in (json.loads(l) for l in (base / "data/processed/sentiment" / (split + ".jsonl")).read_text(encoding="utf-8").splitlines())]
    a, b = load(original), load(module.BASE)
    assert a == b
    checks.append({"split": split, "count": len(a), "exact_text_label_order_match": True})
(ROOT / "运行记录/sentiment_source_check.json").write_text(json.dumps({"status": "passed", "checks": checks}, ensure_ascii=False, indent=2), encoding="utf-8")
print("Default sentiment preparer reproduces the exact experiment data: PASS")
