"""Pin public model revisions after checking the local original model assets."""
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "bert-small-zh": "uer/chinese_roberta_L-2_H-128",
    "t5-small-zh": "uer/t5-small-chinese-cluecorpussmall",
    "qwen05": "Qwen/Qwen2.5-0.5B-Instruct",
    "uie-base-zh": "Pky/uie-base-zh",
}


def get(url):
    response = requests.get(url, timeout=(10, 45))
    response.raise_for_status()
    return response.json()


def main():
    verified = {}
    existing = json.loads((ROOT / "来源版本.json").read_text(encoding="utf-8")).get("models", {})
    for name, repo in MODELS.items():
        sha = existing.get(name, {}).get("revision") or get(f"https://hf-mirror.com/api/models/{repo}")["sha"]
        tree = get(f"https://hf-mirror.com/api/models/{repo}/tree/{sha}?recursive=true&limit=1000")
        assets = []
        for entry in tree:
            path = ROOT / ".cache/models" / name / entry["path"]
            original = entry["path"] in ["config.json", "vocab.txt", "pytorch_model.bin", "model.safetensors"]
            if name == "qwen05":
                original |= entry["path"] in ["tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"]
            if not original or not path.is_file():
                continue
            if name == "bert-small-zh" and entry["path"] == "vocab.txt":
                downloaded = ROOT / ".cache/bert-vocab.txt"
                if downloaded.exists():
                    original_content = downloaded.read_bytes()
                else:
                    response = requests.get(f"https://hf-mirror.com/{repo}/resolve/{sha}/vocab.txt", timeout=(10, 45))
                    response.raise_for_status()
                    original_content = response.content
                assert path.read_text(encoding="utf-8").splitlines() == original_content.decode("utf-8").splitlines()
                actual = hashlib.sha1(f"blob {len(original_content)}\0".encode() + original_content).hexdigest()
                assert actual == entry["oid"]
                assets.append(entry["path"])
                continue
            with path.open("rb") as stream:
                if entry.get("lfs"):
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                    expected = entry["lfs"]["oid"]
                else:
                    content = stream.read()
                    actual = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
                    expected = entry["oid"]
            assert actual == expected, (repo, entry["path"], actual, expected)
            assets.append(entry["path"])
        assert ("config.json" in assets or name == "uie-base-zh") and any(p.endswith((".bin", ".safetensors")) for p in assets)
        verified[name] = {"repository": repo, "revision": sha, "original_assets_verified": assets}
        print(name, sha, len(assets), "assets verified", flush=True)
    path = ROOT / "来源版本.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    source["models"] = verified
    path.write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
