"""Windows downloads for PyG datasets, avoiding implicit framework network calls."""
import argparse
import base64
import json
import pickle
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.experiment import download

BASE = Path(__file__).resolve().parent


def planetoid(name):
    raw = BASE / "task1/data/raw" / name / name / "raw"
    for suffix in ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]:
        filename = f"ind.{name.lower()}.{suffix}"
        target = raw / filename
        if target.exists():
            continue
        api = download(f"https://api.github.com/repos/kimiyoung/planetoid/contents/data/{filename}", raw / (filename + ".json"))
        info = json.loads(api.read_text(encoding="utf-8"))
        if not info.get("content"):
            # The contents endpoint omits payloads larger than 1 MB; blob API retains them.
            blob = download(info["git_url"], raw / (filename + ".blob.json"))
            info = json.loads(blob.read_text(encoding="utf-8"))
        target.write_bytes(base64.b64decode(info["content"]))
        print(target, flush=True)
    from torch_geometric.datasets import Planetoid
    print(Planetoid(str(BASE / "task1/data/raw" / name), name), flush=True)


def flickr():
    from torch_geometric.datasets import Flickr
    raw = BASE / "task1/data/raw/Flickr/raw"
    archive = download("https://data.dgl.ai/dataset/flickr.zip", raw.parent / "flickr.zip", timeout=1200)
    raw.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if Path(name).name in ["adj_full.npz", "feats.npy", "class_map.json", "role.json"]:
                (raw / Path(name).name).write_bytes(z.read(name))
    print(Flickr(str(raw.parent)), flush=True)


def tu():
    raw = BASE / "task3/data/raw/MUTAG/MUTAG/raw"
    archive = download("https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip", BASE / "task3/data/raw/MUTAG.zip")
    raw.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            (raw / Path(name).name).write_bytes(z.read(name))
    from torch_geometric.datasets import TUDataset
    print(TUDataset(str(raw.parent.parent), "MUTAG"), flush=True)


def zinc():
    from scripts.download_model import ranged_download
    folder = BASE / "task3/data/raw/ZINC-HF"
    tree = download("https://hf-mirror.com/api/datasets/graphs-datasets/ZINC/tree/main?recursive=true&limit=1000", ROOT / "zinc-hf-tree.json")
    for entry in json.loads(tree.read_text(encoding="utf-8")):
        if entry["path"].endswith(".parquet"):
            ranged_download("https://hf-mirror.com/datasets/graphs-datasets/ZINC/resolve/main/" + entry["path"],
                            folder / Path(entry["path"]).name, entry["size"], entry.get("lfs", {}).get("oid"))
    print("ZINC official full splits restored from domestic mirror", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["Cora", "CiteSeer", "Flickr", "MUTAG", "ZINC"])
    args = p.parse_args()
    if args.dataset in ["Cora", "CiteSeer"]:
        planetoid(args.dataset)
    else:
        {"Flickr": flickr, "MUTAG": tu, "ZINC": zinc}[args.dataset]()
