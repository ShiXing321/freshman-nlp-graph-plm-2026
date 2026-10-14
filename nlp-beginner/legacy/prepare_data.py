"""Fetch real datasets, keeping published train/dev/test boundaries."""
import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.experiment import download

BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"


def sentiment(kaggle=None):
    if kaggle:
        import random
        rows = list(csv.DictReader(Path(kaggle).open(encoding="utf-8"), delimiter="\t"))
        groups = sorted({r["SentenceId"] for r in rows})
        random.Random(42).shuffle(groups)
        boundaries = {g: ("train" if i < int(len(groups) * .8) else "dev" if i < int(len(groups) * .9) else "test") for i, g in enumerate(groups)}
        result = {s: [] for s in ["train", "dev", "test"]}
        for r in rows:
            result[boundaries[r["SentenceId"]]].append({"text": r["Phrase"], "label": int(r["Sentiment"]), "sentence_id": r["SentenceId"]})
        path = Path(kaggle).resolve()
        source = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name
        return write("sentiment_kaggle", result, {"source": source, "variant": "Kaggle phrase-level data, SentenceId grouped 80/10/10 split; seed=42"})
    result = {}
    for split in ["train", "dev", "test"]:
        path = download(f"https://hf-mirror.com/datasets/SetFit/sst5/resolve/main/{split}.jsonl", RAW / "sst5" / f"{split}.jsonl", timeout=600)
        result[split] = [{"text": r["text"], "label": int(r["label"])} for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l)]
    return write("sentiment", result, {"source": "SetFit/sst5 mirror of Stanford Sentiment Treebank", "variant": "official sentence-level split; not Kaggle phrase-level dataset"})


def sentiment_stanford_archive():
    archive = download("https://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip", RAW / "sst.zip")
    with zipfile.ZipFile(archive) as z:
        def read(name):
            return z.read("stanfordSentimentTreebank/" + name).decode("utf-8").splitlines()
        scores = dict(line.split("|") for line in read("sentiment_labels.txt")[1:])
        phrases = {phrase: int(pid) for phrase, pid in (line.rsplit("|", 1) for line in read("dictionary.txt"))}
        splits = dict(line.split(",") for line in read("datasetSplit.txt")[1:])
        result = {"train": [], "dev": [], "test": []}
        for line in read("datasetSentences.txt")[1:]:
            sid, text = line.split("\t", 1)
            text = text.replace("-LRB-", "(").replace("-RRB-", ")")
            pid = phrases.get(text)
            if pid is None:
                raise ValueError(f"SST sentence {sid} missing label")
            value = float(scores[str(pid)])
            label = min(4, int(value * 5))
            split = {"1": "train", "2": "test", "3": "dev"}[splits[sid]]
            result[split].append({"text": text, "label": label, "sentence_id": int(sid)})
    write("sentiment", result, {"source": "Stanford Sentiment Treebank (same source as Rotten Tomatoes)",
                                "variant": "official sentence-level split, not Kaggle phrase-level train.tsv"})


def hf_parquet(dataset, prefix, fields):
    import pandas as pd
    api = download(f"https://hf-mirror.com/api/datasets/{dataset}/tree/main?recursive=true&limit=1000", RAW / (prefix + "-tree.json"))
    entries = json.loads(api.read_text(encoding="utf-8"))
    revision = "main"
    if not any(x["path"].endswith(".parquet") for x in entries):
        revision = "refs%2Fconvert%2Fparquet"
        api = download(f"https://hf-mirror.com/api/datasets/{dataset}/tree/{revision}?recursive=true&limit=1000", RAW / (prefix + "-converted-tree.json"))
        entries = json.loads(api.read_text(encoding="utf-8"))
    result = {"train": [], "dev": [], "test": []}
    for split, key in (("train", "train"), ("validation", "dev"), ("test", "test")):
        candidates = [x["path"] for x in entries if x["path"].endswith(".parquet") and (f"/{split}-" in x["path"] or f"/{split}/" in x["path"])]
        if not candidates:
            raise RuntimeError(f"No parquet found for {dataset}: {split}")
        for path in candidates:
            file = download(f"https://hf-mirror.com/datasets/{dataset}/resolve/{revision}/{path}", RAW / prefix / (split + "-" + Path(path).name), timeout=1200)
            for row in pd.read_parquet(file).to_dict("records"):
                value = fields(row)
                if value is not None:
                    result[key].append(value)
    write(prefix, result, {"source": dataset, "variant": "official splits"})


def write(name, splits, metadata):
    folder = BASE / "data" / "processed" / name
    folder.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        (folder / f"{split}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    metadata["counts"] = {k: len(v) for k, v in splits.items()}
    (folder / "source.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(name, metadata, flush=True)


def snli():
    hf_parquet("stanfordnlp/snli", "snli", lambda r: {"premise": r["premise"], "hypothesis": r["hypothesis"], "label": int(r["label"])} if int(r["label"]) >= 0 else None)


def ner():
    hf_parquet("eriktks/conll2003", "ner", lambda r: {"tokens": list(r["tokens"]), "tags": [int(t) for t in r["ner_tags"]]})


def poetry():
    source = BASE.parent / "poetryFromTang.txt"
    text = source.read_text(encoding="utf-8-sig")
    # Separate contiguous regions before making windows; no random overlapping-window split.
    n = len(text)
    write("poetry", {"train": [{"text": text[:int(n * .8)]}],
                     "dev": [{"text": text[int(n * .8):int(n * .9)]}],
                     "test": [{"text": text[int(n * .9):]}]}, {"source": str(source.relative_to(ROOT)), "variant": "contiguous 80/10/10 split"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["sentiment", "snli", "ner", "poetry", "all"])
    parser.add_argument("--kaggle-tsv", help="Optional Kaggle train.tsv; grouped by SentenceId")
    args = parser.parse_args()
    for name, fn in [("sentiment", sentiment), ("snli", snli), ("ner", ner), ("poetry", poetry)]:
        if args.dataset in (name, "all"):
            fn(args.kaggle_tsv) if name == "sentiment" else fn()
