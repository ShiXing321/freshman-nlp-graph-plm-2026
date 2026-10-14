"""Compare optimized filtered ranking against direct candidate scoring."""
import importlib.util
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "graph_beginner/task4/code"))
spec = importlib.util.spec_from_file_location("kge_train", ROOT / "graph_beginner/task4/code/train.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.manual_seed(42)
triples = torch.tensor([[0, 0, 2], [1, 0, 2], [2, 1, 4], [3, 1, 4]])
known = set(map(tuple, triples.tolist()))
checks = []
for cls in [module.TransE, module.RotatE, module.ConvE]:
    model = cls(11, 2, 64).eval()
    actual = module.evaluate(model, triples, 11, known, 0, torch.device("cpu"))
    ranks = []
    candidates = torch.arange(11)
    with torch.no_grad():
        for h, r, t in triples.tolist():
            for side in ["head", "tail"]:
                scores = model(candidates if side == "head" else torch.full_like(candidates, h),
                               torch.full_like(candidates, r),
                               candidates if side == "tail" else torch.full_like(candidates, t))
                correct = h if side == "head" else t
                gold = scores[correct].clone()
                others = [a if side == "head" else c for a, b, c in known
                          if (b == r and c == t if side == "head" else a == h and b == r)]
                scores[others] = -torch.inf
                scores[correct] = gold
                ranks.append(float((scores > gold).sum()) + 1 + .5 * (int((scores == gold).sum()) - 1))
    expected = float((1 / torch.tensor(ranks, dtype=torch.float)).mean())
    assert abs(actual["mrr"] - expected) < 1e-7, (cls.__name__, actual, ranks)
    checks.append({"model": cls.__name__, "passed": True, "mrr": expected})
out = ROOT / "运行记录/核查/kge_ranking.json"
out.write_text(json.dumps({"status": "passed", "checks": checks}, indent=2), encoding="utf-8")
print(json.dumps(checks, indent=2))
