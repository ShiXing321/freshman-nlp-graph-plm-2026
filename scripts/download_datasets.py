"""Resume the remaining HF dataset files, validating their published LFS hashes."""
import argparse
import json
from pathlib import Path
from download_model import ranged_download

ROOT = Path(__file__).resolve().parents[1]

def main(dataset):
    if dataset == "snli":
        ranged_download("https://hf-mirror.com/datasets/stanfordnlp/snli/resolve/main/plain_text/train-00000-of-00001.parquet",
                        ROOT / "nlp-beginner/legacy/data/raw/snli/train-train-00000-of-00001.parquet", 19614612,
                        "ef9a7b25d97390a62aeda7abe26aec8640600f50b818eaeb9107097d60ac6620")
    else:
        entries = json.loads((ROOT / "zinc-hf-tree.json").read_text())
        for entry in entries:
            if entry["path"].endswith(".parquet"):
                ranged_download("https://hf-mirror.com/datasets/graphs-datasets/ZINC/resolve/main/" + entry["path"],
                                ROOT / "graph_beginner/task3/data/raw/ZINC-HF" / Path(entry["path"]).name,
                                entry["size"], entry["lfs"]["oid"])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["snli", "zinc"])
    main(p.parse_args().dataset)
