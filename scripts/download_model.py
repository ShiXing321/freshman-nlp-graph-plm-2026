"""Download HF model files using a mirror, size checks, and LFS SHA-256 verification."""
import argparse
import concurrent.futures
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.experiment import download


def ranged_download(url, destination, size, sha=None, workers=4, source_offset=0, chunk_size=8 * 1024 * 1024):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == size:
        with destination.open("rb") as stream:
            valid = not sha or hashlib.file_digest(stream, "sha256").hexdigest() == sha
        if valid:
            return destination
    chunk = chunk_size
    parts = destination.parent / (destination.name + ".chunks")
    parts.mkdir(exist_ok=True)
    def fetch(i):
        begin, end = source_offset + i * chunk, source_offset + min(size, (i + 1) * chunk) - 1
        path = parts / f"{i:04d}.part"
        if path.exists() and path.stat().st_size > end - begin + 1:
            raise ValueError(f"Oversized existing range: {path}")
        for attempt in range(30):
            have = path.stat().st_size if path.exists() else 0
            if have == end - begin + 1:
                break
            more, headers = parts / f"{i:04d}.more", parts / f"{i:04d}.headers"
            result = subprocess.run(["curl.exe", "-sSfL", "--connect-timeout", "15", "--max-time", "180", "--speed-limit", "1000", "--speed-time", "30",
                                     "--range", f"{begin + have}-{end}", url, "-D", str(headers), "-o", str(more)])
            # Retain a correctly identified partial transfer even after a timeout.
            ranges = re.findall(r"content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)", headers.read_text(encoding="utf-8", errors="replace") if headers.exists() else "", flags=re.I)
            if more.exists() and ranges and int(ranges[-1][0]) == begin + have and more.stat().st_size <= end - begin + 1 - have:
                with path.open("ab") as out, more.open("rb") as incoming:
                    while block := incoming.read(1024 * 1024):
                        out.write(block)
            elif result.returncode == 0:
                raise ValueError(f"Server ignored requested range: {path}")
        if path.stat().st_size != end - begin + 1:
            raise ValueError(f"Server ignored range: {path}")
        print(destination.name, i, "complete", flush=True)
        return path
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        paths = list(pool.map(fetch, range((size + chunk - 1) // chunk)))
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    with temporary.open("wb") as out:
        for path in paths:
            with path.open("rb") as f:
                while block := f.read(1024 * 1024):
                    digest.update(block)
                    out.write(block)
    if sha and digest.hexdigest() != sha:
        raise ValueError(f"SHA-256 mismatch for {destination.name}")
    temporary.replace(destination)
    return destination


def model(repo, folder, revision=None):
    folder = Path(folder)
    if revision is None:
        sources = json.loads((ROOT / "来源版本.json").read_text(encoding="utf-8"))
        revision = next((row["revision"] for row in sources.get("models", {}).values() if row["repository"] == repo), "main")
    tree_path = download(f"https://hf-mirror.com/api/models/{repo}/tree/{revision}?recursive=true&limit=1000", folder / f"files-{revision}.json")
    (folder / "files.json").write_bytes(tree_path.read_bytes())
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    safetensors = any(x["path"].endswith(".safetensors") for x in tree)
    needed = [x for x in tree if x["path"].endswith((".json", ".txt", ".model")) or (x["path"].endswith(".safetensors") if safetensors else x["path"] == "pytorch_model.bin")]
    for entry in sorted(needed, key=lambda x: x["size"]):
        path = entry["path"]
        if path in ["README.md", "files.json"]:
            continue
        url = f"https://hf-mirror.com/{repo}/resolve/{revision}/{path}"
        if entry["size"] > 8 * 1024 * 1024:
            ranged_download(url, folder / path, entry["size"], entry.get("lfs", {}).get("oid"))
        else:
            download(url, folder / path, timeout=600)
    print(repo, "downloaded", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("repo")
    p.add_argument("folder")
    args = p.parse_args()
    model(args.repo, args.folder)
