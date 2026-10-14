"""Audit source copies, data/model hashes, math checks and saved checkpoints."""
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def main():
    out = ROOT / "运行记录/核查"
    out.mkdir(parents=True, exist_ok=True)
    report = {"time": datetime.now().astimezone().isoformat(), "checks": []}
    def record(name, passed, **details):
        report["checks"].append({"name": name, "passed": bool(passed), **details})
        print(name, "PASS" if passed else "FAIL", flush=True)
    env = os.environ | {"PYTHONUTF8": "1", "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"}
    for name, command in [
        ("syntax", ["-m", "compileall", "-q", "common", "scripts", "nlp-beginner/legacy", "graph_beginner/task1/code", "graph_beginner/task2/code", "graph_beginner/task3/code", "graph_beginner/task4/code", "transformers_tasks/local_runs"]),
        ("numpy_crf_math", ["scripts/verify_algorithms.py"]),
        ("ppo_math", ["scripts/verify_plm_math.py"]),
    ]:
        proc = subprocess.run([sys.executable, *command], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        (out / (name + ".txt")).write_text(proc.stdout + proc.stderr, encoding="utf-8")
        record(name, proc.returncode == 0, log=str((out / (name + ".txt")).relative_to(ROOT)))
    tree = json.loads((ROOT / "plm-tree.json").read_text(encoding="utf-8"))
    tree = tree["tree"] if isinstance(tree, dict) else tree
    import zipfile
    upstream_archive = ROOT / ".cache/upstream/transformers_tasks.zip"
    archive_contents = {}
    if upstream_archive.exists():
        with zipfile.ZipFile(upstream_archive) as source:
            for entry in source.infolist():
                if not entry.is_dir():
                    relative = entry.filename.split("/", 1)[1]
                    archive_contents[relative] = source.read(entry)
    failures, checked = [], 0
    for entry in tree:
        if entry.get("type") != "blob":
            continue
        path = ROOT / "transformers_tasks" / entry["path"]
        if entry["path"] in archive_contents:
            data = archive_contents[entry["path"]]
        elif path.exists():
            data = path.read_bytes()
        else:
            failures.append(entry["path"] + " missing")
            continue
        actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if actual != entry["sha"]:
            failures.append(entry["path"] + " hash differs")
        checked += 1
    record("PLM upstream Git blobs", not failures, files_checked=checked, failures=failures)
    for folder in ["qwen05", "t5-small-zh"]:
        directory = ROOT / ".cache/models" / folder
        for entry in json.loads((directory / "files.json").read_text(encoding="utf-8")):
            if entry.get("lfs") and entry["path"].endswith((".bin", ".safetensors")):
                path = directory / entry["path"]
                actual = sha256(path) if path.exists() else None
                record(folder + " weights SHA256", actual == entry["lfs"]["oid"], actual=actual, expected=entry["lfs"]["oid"])
    for entry in json.loads((ROOT / "zinc-hf-tree.json").read_text(encoding="utf-8")):
        if entry["path"].endswith(".parquet"):
            path = ROOT / "graph_beginner/task3/data/raw/ZINC-HF" / Path(entry["path"]).name
            actual = sha256(path) if path.exists() else None
            record("ZINC " + path.name, actual == entry["lfs"]["oid"], sha256=actual)
    path = ROOT / "nlp-beginner/legacy/data/raw/snli/train-train-00000-of-00001.parquet"
    record("SNLI training parquet", path.exists() and sha256(path) == "ef9a7b25d97390a62aeda7abe26aec8640600f50b818eaeb9107097d60ac6620")
    import zlib
    path = ROOT / "nlp-beginner/legacy/data/raw/glove/glove.6B.50d.txt"
    crc = 0
    with path.open("rb") as f:
        while block := f.read(1024 * 1024):
            crc = zlib.crc32(block, crc)
    record("GloVe official ZIP entry CRC", crc == 3530997057 and path.stat().st_size == 171350515, crc=crc)
    import zipfile
    with zipfile.ZipFile(ROOT / "graph_beginner/task1/data/raw/Flickr/flickr.zip") as z:
        invalid = z.testzip()
    record("Flickr official ZIP CRC", invalid is None, invalid_entry=invalid)
    import torch
    torch.set_num_threads(2)
    checkpoints = list((ROOT / "nlp-beginner/legacy").glob("task*/checkpoints/*.pt")) + list((ROOT / "graph_beginner").glob("task*/checkpoints/*.pt")) + list((ROOT / "transformers_tasks/local_runs/checkpoints").glob("*/model.pt")) + list((ROOT / "transformers_tasks/local_runs/checkpoints").glob("*/value_head.pt"))
    nonfinite = []
    for path in checkpoints:
        # These are our own locally written checkpoints; some contain NumPy metadata.
        state = torch.load(path, map_location="cpu", weights_only=False)
        state = state.get("state", state)
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
                nonfinite.append(str(path.relative_to(ROOT)) + ":" + key)
    record("checkpoint tensors finite", not nonfinite, checkpoints=len(checkpoints), failures=nonfinite)
    from safetensors import safe_open
    tensor_files = list((ROOT / "transformers_tasks/local_runs/checkpoints").rglob("*.safetensors"))
    nonfinite = []
    for path in tensor_files:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for name in f.keys():
                if not torch.isfinite(f.get_tensor(name)).all():
                    nonfinite.append(str(path.relative_to(ROOT)) + ":" + name)
    record("saved generation tensors finite", not nonfinite, files=len(tensor_files), failures=nonfinite)
    # Independent checkpoint restore and forward pass on real-like valid padded inputs.
    sys.path.insert(0, str(ROOT / "nlp-beginner/legacy"))
    from models import TextClassifier, ESIM, BiLSTMCRF, CharLM
    tested = []
    for path in (ROOT / "nlp-beginner/legacy").glob("task*/checkpoints/*.pt"):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if "vocab" not in state:
            continue
        c, n = state["config"], len(state["vocab"])
        task = c["task"]
        model = TextClassifier(n, c["model"], c["dim"], c["hidden"]) if task == "classification" else ESIM(n, c["dim"], c["hidden"]) if task == "matching" else BiLSTMCRF(n, dim=c["dim"], hidden=c["hidden"], use_case=c.get("use_case", False)) if task == "ner" else CharLM(n, c["model"], c["dim"], c["hidden"])
        model.load_state_dict(state["state"])
        model.eval()
        ids = torch.tensor([[2, 3, 4, 0], [3, 2, 0, 0]])
        with torch.no_grad():
            prediction = model(ids, ids) if task == "matching" else model(ids, ids.ne(0).long()) if task == "ner" and c.get("use_case") else model(ids)
        if task == "lm":
            prediction = prediction[0]
        assert torch.isfinite(prediction).all()
        if task == "ner":
            decoded = model.crf.decode(prediction, ids.ne(0))
            assert [len(x) for x in decoded] == [3, 2]
        tested.append(str(path.relative_to(ROOT)))
    record("NLP checkpoint forward", True, models=tested)
    import importlib.util
    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    graph_models = module("graph_models_audit", ROOT / "graph_beginner/models.py")
    kg_models = module("kg_models_audit", ROOT / "graph_beginner/task4/code/models.py")
    from torch_geometric.data import Data
    edges = torch.tensor([[0, 1, 1, 2, 2, 3, 5, 6, 6, 7, 7, 8], [1, 0, 2, 1, 3, 2, 6, 5, 7, 6, 8, 7]])
    tested = []
    dimensions = {"Cora": (1433, 7), "CiteSeer": (3703, 6), "Flickr": (500, 7)}
    for path in (ROOT / "graph_beginner").glob("task*/checkpoints/*.pt"):
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        c = snapshot["config"]
        task = path.parents[1].name
        if task in ["task1", "task2"]:
            features, classes = dimensions[c["dataset"]]
            model = graph_models.GNN(features, classes if task == "task1" else c["hidden"], c["model"], c["hidden"], c["layers"])
            inputs = (torch.randn(10, features), edges)
        elif task == "task3":
            regression = c["dataset"] == "ZINC"
            model = graph_models.GraphPredictor(32 if regression else 7, 1 if regression else 2, c["model"], c["pool"], c["hidden"], c["layers"], 28 if regression else 0)
            x = torch.randint(28, (10, 1)) if regression else torch.randn(10, 7)
            inputs = (Data(x=x, edge_index=edges, batch=torch.tensor([0]*5 + [1]*5)),)
        else:
            model = {"transe": kg_models.TransE, "rotate": kg_models.RotatE, "conve": kg_models.ConvE}[c["model"]](len(snapshot["entities"]), len(snapshot["relations"]), c["dim"])
            inputs = (torch.tensor([0, 1]), torch.tensor([0, 0]), torch.tensor([1, 2]))
        model.load_state_dict(snapshot["state"])
        model.eval()
        with torch.no_grad():
            prediction = model(*inputs)
        assert torch.isfinite(prediction).all()
        tested.append(str(path.relative_to(ROOT)))
    record("Graph checkpoint forward", True, models=tested)
    # Group split audits use the actual loader, without GPU or training.
    sys.path.insert(0, str(ROOT / "transformers_tasks/local_runs"))
    from bert_tasks import load_rows
    from bert_tasks import prepare_model, BertTask
    pretrained, tok = prepare_model()
    tested = []
    for name in ["classification", "pointwise", "dssm", "sbert", "simcse", "pet", "ptuning"]:
        path = ROOT / "transformers_tasks/local_runs/checkpoints" / name / "model.pt"
        if not path.exists():
            continue
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        model = BertTask(pretrained, tok, name, snapshot["config"]["labels"])
        model.load_state_dict(snapshot["state"])
        model.eval()
        sample, _ = load_rows(name, 42)
        with torch.no_grad():
            logits = model(sample["test"][:2], torch.device("cpu"))[1]
        assert torch.isfinite(logits).all()
        tested.append(name)
    record("BERT checkpoint forward", True, models=tested)
    from span_task import SpanModel, encode, data as span_data
    tested = []
    for arm in ["none", "negative", "swap"]:
        snapshot = torch.load(ROOT / "transformers_tasks/local_runs/checkpoints" / ("uie_" + arm) / "model.pt", map_location="cpu", weights_only=False)
        model = SpanModel(pretrained)
        model.load_state_dict(snapshot["state"])
        model.eval()
        inputs, *_ = encode(tok, span_data(42)["test"][:2], torch.device("cpu"))
        with torch.no_grad():
            output = model(inputs)
        assert torch.isfinite(output).all() and output.shape[-1] == 2
        tested.append(arm)
    record("UIE checkpoint forward", True, models=tested)
    from span_task import prepare_span
    uie_path, uie_tok = prepare_span(True)
    tested = []
    for arm in ["none", "negative", "swap", "fill"]:
        path = ROOT / "transformers_tasks/local_runs/checkpoints" / ("uie_base_" + arm) / "model.pt"
        snapshot = torch.load(path, map_location="cpu", weights_only=True)
        model = SpanModel(uie_path).eval()
        model.load_state_dict(snapshot["state"])
        inputs, *_ = encode(uie_tok, span_data(42)["test"][:2], torch.device("cpu"))
        with torch.no_grad():
            output = model(inputs)
        assert torch.isfinite(output).all() and output.shape[-1] == 2
        tested.append(arm)
        del snapshot, model
    record("Pretrained UIE checkpoint forward", True, models=tested)
    from reward_task import RewardModel
    model = RewardModel(pretrained)
    snapshot = torch.load(ROOT / "transformers_tasks/local_runs/checkpoints/reward/model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(snapshot["state"])
    model.eval()
    with torch.no_grad():
        scores = model(tok(["味道很好，包装也不错。", "东西坏了，无法使用。"], return_tensors="pt", padding=True))
    record("reward checkpoint forward", bool(torch.isfinite(scores).all()), examples=scores.tolist())
    for task in ["pointwise", "dssm", "sbert"]:
        rows, _ = load_rows(task, 42)
        keys = {s: {r["b"] for r in values} for s, values in rows.items()}
        overlaps = {a + "/" + b: len(keys[a] & keys[b]) for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]}
        record(task + " grouped splits", not any(overlaps.values()), overlaps=overlaps)
    from span_task import data
    rows = data(42)
    keys = {s: {r["content"] for r in v} for s, v in rows.items()}
    record("UIE grouped splits", all(not keys[a] & keys[b] for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]))
    from llm_tasks import pretrain_rows, sft_rows
    for name, rows, key in [("scratch docs", pretrain_rows(1000, 42), "text"), ("Qwen SFT prompts", sft_rows(400, 42), "input")]:
        keys = {s: {r[key] for r in v} for s, v in rows.items()}
        record(name + " independent holdouts", all(not keys[a] & keys[b] for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]))
    report["status"] = "passed" if all(c["passed"] for c in report["checks"]) else "failed"
    (out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report["status"], flush=True)
    return report["status"] == "passed"

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
