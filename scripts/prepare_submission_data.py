"""Restore training inputs and pretrained models using public mirrors."""
import argparse
import base64
import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.experiment import download
from download_model import model


def plm_data():
    sha = "5464a2c16b73d1c8382f40fe48682830a87dfe65"
    archive = ROOT / ".cache/upstream/transformers_tasks.zip"
    urls = [f"https://api.github.com/repos/HarderThenHarder/transformers_tasks/zipball/{sha}",
            f"https://ghproxy.net/https://github.com/HarderThenHarder/transformers_tasks/archive/{sha}.zip"]
    for url in urls:
        try:
            download(url, archive, timeout=1200)
            with zipfile.ZipFile(archive) as source:
                assert source.testzip() is None
            break
        except Exception as error:
            print("upstream archive", type(error).__name__, flush=True)
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            relative = PurePosixPath(item.filename)
            if item.is_dir() or len(relative.parts) < 2:
                continue
            tail = PurePosixPath(*relative.parts[1:])
            if ".." in tail.parts or tail.is_absolute():
                raise ValueError("Invalid archive path")
            if "data" not in tail.parts:
                continue
            target = ROOT / "transformers_tasks" / Path(*tail.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read(item))
    print("PLM upstream data restored", flush=True)


def kaggle_data():
    import importlib.util
    folder = ROOT / "nlp-beginner/legacy/data/raw/kaggle"
    path = folder / "train.tsv"
    if not path.exists():
        response = download("https://api.github.com/repos/vivanraaj/Sentiment-Analysis-on-Movie-Reviews/git/blobs/35f3b73e9cfb1e2634fcf35182ea2584de7b88cf", folder / "train.blob.json")
        info = json.loads(response.read_text(encoding="utf-8"))
        path.write_bytes(base64.b64decode(info["content"]))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "b59d3b8ab798d0d4f2469932e32ebfa89c9efd0dd2989aa34437c93004974bee"
    (folder / "source.json").write_text(json.dumps({
        "dataset": "Kaggle sentiment-analysis-on-movie-reviews", "rows": 156060,
        "repository": "vivanraaj/Sentiment-Analysis-on-Movie-Reviews",
        "blob": "35f3b73e9cfb1e2634fcf35182ea2584de7b88cf",
        "sha256": "b59d3b8ab798d0d4f2469932e32ebfa89c9efd0dd2989aa34437c93004974bee",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("nlp_prepare", ROOT / "nlp-beginner/legacy/prepare_data.py")
    prepared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepared)
    prepared.sentiment(str(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=["nlp", "graph", "plm", "models", "uie", "all"])
    args = parser.parse_args()
    if args.group in ["plm", "all"]:
        plm_data()
    if args.group in ["nlp", "all"]:
        kaggle_data()
        import subprocess
        for dataset in ["snli", "ner", "poetry"]:
            subprocess.run([sys.executable, "nlp-beginner/legacy/prepare_data.py", dataset], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "scripts/download_glove50.py"], cwd=ROOT, check=True)
    if args.group in ["graph", "all"]:
        import subprocess
        for dataset in ["Cora", "CiteSeer", "Flickr", "MUTAG", "ZINC"]:
            subprocess.run([sys.executable, "graph_beginner/prepare_data.py", dataset], cwd=ROOT, check=True)
    if args.group in ["models", "all"]:
        for repo, folder in [("uer/chinese_roberta_L-2_H-128", "bert-small-zh"),
                             ("uer/t5-small-chinese-cluecorpussmall", "t5-small-zh"),
                             ("Qwen/Qwen2.5-0.5B-Instruct", "qwen05")]:
            model(repo, ROOT / ".cache/models" / folder)
    if args.group in ["uie", "all"]:
        import subprocess
        model("Pky/uie-base-zh", ROOT / ".cache/models/uie-base-zh")
        subprocess.run([sys.executable, "scripts/convert_uie_checkpoint.py"], cwd=ROOT, check=True)
