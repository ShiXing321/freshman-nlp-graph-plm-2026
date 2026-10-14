"""Compare CPU and CUDA scores from the same saved Cora link models."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "graph_beginner"), str(ROOT / "graph_beginner/task1/code")]
from common.experiment import seed_all
from train import dataset
from models import GNN


def scores(model, data, device):
    model.to(device).eval()
    graph = data.clone().to(device)
    with torch.no_grad():
        z = model(graph.x, graph.edge_index)
        value = (z[graph.edge_label_index[0]] * z[graph.edge_label_index[1]]).sum(-1)
    return value.cpu().numpy()


def main():
    assert torch.cuda.is_available()
    ds = dataset("Cora")
    records = []
    for kind in ["gcn", "gat", "sage", "gin"]:
        seed_all(42)
        _, _, data = RandomLinkSplit(num_val=.1, num_test=.1, is_undirected=True, add_negative_train_samples=True,
            neg_sampling_ratio=1., disjoint_train_ratio=.2)(ds[0])
        name = f"Cora_{kind}_full_l2_lr0.01_isolated"
        path = ROOT / "graph_beginner/task2/checkpoints" / (name + ".pt")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        c = saved["config"]
        model = GNN(ds.num_features, c["hidden"], kind, c["hidden"], c["layers"])
        model.load_state_dict(saved["state"])
        cpu, gpu = scores(model, data, "cpu"), scores(model, data, "cuda")
        labels = data.edge_label.numpy()
        def evaluate(value):
            return {"auc": roc_auc_score(labels, value), "average_precision": average_precision_score(labels, value)}
        pos, neg = labels == 1, labels == 0
        changed = int((np.sign(cpu[pos, None] - cpu[neg][None, :]) != np.sign(gpu[pos, None] - gpu[neg][None, :])).sum())
        numerical = np.allclose(cpu, gpu, rtol=2e-5, atol=1e-6)
        row = {"model": kind, "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "test_edges_sha256": hashlib.sha256(data.edge_label_index.numpy().tobytes() + labels.tobytes()).hexdigest(),
            "test_positive": int(pos.sum()), "test_negative": int(neg.sum()), "max_score_absolute_difference": float(np.abs(cpu - gpu).max()),
            "numerically_close": bool(numerical), "positive_negative_order_changes": changed,
            "cpu": evaluate(cpu), "cuda": evaluate(gpu),
            "recorded_cuda": json.loads((ROOT / "graph_beginner/task2/results" / (name + ".json")).read_text(encoding="utf-8"))["test"]}
        records.append(row)
        print(json.dumps(row), flush=True)
    result = {"status": "passed" if all(r["numerically_close"] for r in records) else "failed", "records": records}
    (ROOT / "运行记录/诊断/link_reload_devices.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result["status"] == "passed"


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
