"""Exercise local read endpoints and reject an invalid rating without writing one."""
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
checks = []
for model in ["bert", "qwen", "scratch"]:
    query = urlencode({"model": model, "text": "我在学习NLP。Hello, world!"})
    value = json.loads(urlopen("http://127.0.0.1:8765/api/tokenize?" + query, timeout=60).read())
    assert value["count"] == len(value["ids"]) == len(value["tokens"]) == len(value["offsets"]) and value["count"] > 0
    checks.append({"model": model, "result": value})
value = json.loads(urlopen("http://127.0.0.1:8765/api/samples").read())
assert value["name"].startswith("qwen_ppo_v2") and len(value["samples"]) > 0
rating_path = ROOT / "transformers_tasks/local_runs/human_ratings.jsonl"
size = rating_path.stat().st_size if rating_path.exists() else 0
try:
    request = Request("http://127.0.0.1:8765/api/rating", data=json.dumps({"index": 0, "score": 0}).encode(), headers={"Content-Type": "application/json"}, method="POST")
    urlopen(request)
    raise AssertionError("Invalid rating accepted")
except HTTPError as e:
    assert e.code == 400
assert (rating_path.stat().st_size if rating_path.exists() else 0) == size
report = {"status": "passed", "tokenizers": checks, "ppo_result": value["name"], "ppo_sample_count": len(value["samples"]), "invalid_rating_rejected": True, "human_ratings_added_by_test": 0}
(ROOT / "运行记录/local_tools_checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("Local tokenizer endpoints and rating validation: PASS")
