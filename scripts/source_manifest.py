"""Record reproducible data/model provenance without copying the large files."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def main():
    entries = []
    sources = json.loads((ROOT / "来源版本.json").read_text(encoding="utf-8")).get("models", {})
    def add(path, source, metadata=None):
        path = ROOT / path
        if not path.exists():
            return
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        entries.append({"file": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                        "sha256": digest, "source": source, "metadata": metadata or {}})
    for name, repo in [("bert-small-zh", "uer/chinese_roberta_L-2_H-128"), ("t5-small-zh", "uer/t5-small-chinese-cluecorpussmall"),
                       ("qwen05", "Qwen/Qwen2.5-0.5B-Instruct"), ("uie-base-zh", "Pky/uie-base-zh")]:
        folder = ROOT / ".cache/models" / name
        for path in folder.glob("*"):
            if path.is_file() and path.suffix in [".bin", ".safetensors", ".json", ".txt", ".model"]:
                metadata = {"revision": sources.get(name, {}).get("revision", "main")}
                if name == "bert-small-zh" and path.name in ["vocab.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
                    metadata["processing"] = "BertTokenizerFast saved locally with do_lower_case=True; vocabulary token order retained"
                add(path.relative_to(ROOT), "https://hf-mirror.com/" + repo, metadata)
    path = ROOT / "nlp-beginner/legacy/data/raw/kaggle/source.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    add("nlp-beginner/legacy/data/raw/kaggle/train.tsv", "Kaggle sentiment-analysis-on-movie-reviews public unchanged copy", metadata)
    add("nlp-beginner/legacy/data/raw/glove/glove.6B.50d.txt", "Stanford GloVe 6B official ZIP 50d entry", {"zip_crc": 3530997057})
    add("graph_beginner/task1/data/raw/Flickr/flickr.zip", "https://data.dgl.ai/dataset/flickr.zip")
    add("nlp-beginner/poetryFromTang.txt", "FudanNLP/nlp-beginner original poetry corpus")
    for name in ["Cora", "CiteSeer"]:
        folder = ROOT / "graph_beginner/task1/data/raw" / name / name / "raw"
        for path in folder.glob("ind.*"):
            if path.is_file() and path.suffix != ".json":
                add(path.relative_to(ROOT), "https://github.com/kimiyoung/planetoid/tree/master/data")
    folder = ROOT / "graph_beginner/task3/data/raw/MUTAG/MUTAG/raw"
    for path in folder.glob("*.txt"):
        add(path.relative_to(ROOT), "https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip")
    for path in (ROOT / "graph_beginner/task4/data/raw/FB15k-237").glob("*.txt"):
        add(path.relative_to(ROOT), "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/tree/master/data/FB15k-237")
    for folder, source in [("nlp-beginner/legacy/data/raw/snli", "stanfordnlp/snli"),
                           ("nlp-beginner/legacy/data/raw/ner", "eriktks/conll2003"),
                           ("graph_beginner/task3/data/raw/ZINC-HF", "graphs-datasets/ZINC")]:
        for path in (ROOT / folder).glob("*.parquet"):
            add(path.relative_to(ROOT), "https://hf-mirror.com/datasets/" + source)
    for path in (ROOT / "nlp-beginner/legacy/data/processed").glob("*/source.json"):
        add(path.relative_to(ROOT), "processed data split metadata", json.loads(path.read_text(encoding="utf-8")))
    output = ROOT / "运行记录/数据模型来源.json"
    output.write_text(json.dumps({"files": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Source manifest", len(entries), flush=True)


if __name__ == "__main__":
    main()
