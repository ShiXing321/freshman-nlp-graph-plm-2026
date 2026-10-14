"""Fetch just the 50d entry of the official GloVe ZIP, verifying its ZIP CRC."""
import concurrent.futures
import struct
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.experiment import download

URL = "https://hf-mirror.com/stanfordnlp/glove/resolve/main/glove.6B.zip"
OFFSET = 792999805
COMPRESSED = 69182505
UNCOMPRESSED = 171350515
folder = ROOT / "nlp-beginner/legacy/data/raw/glove"
folder.mkdir(parents=True, exist_ok=True)


def fetch(i):
    # ZIP local header is 74 bytes: 30 base + 16 name + 28 extra.
    begin = OFFSET + 74 + i * 4 * 1024 * 1024
    end = min(OFFSET + 74 + COMPRESSED, begin + 4 * 1024 * 1024) - 1
    path = folder / f"range-{i:03d}.part"
    if not path.exists() or path.stat().st_size != end - begin + 1:
        subprocess.run(["curl.exe", "-sSfL", "--retry", "2", "--connect-timeout", "15", "--max-time", "600", "--range", f"{begin}-{end}", URL, "-o", str(path)], check=True)
    if path.stat().st_size != end - begin + 1:
        raise ValueError("GloVe server ignored range")
    print("range", i, "complete", flush=True)
    return path


if __name__ == "__main__":
    # Read actual local header rather than relying on an unchecked fixed offset.
    header = folder / "header.part"
    expected_crc = 3530997057  # Recorded from the official ZIP central directory.
    if header.exists() and header.stat().st_size >= 74:
        h = struct.unpack_from("<IHHHHHIIIHH", header.read_bytes())
        assert h[0] == 0x04034b50 and h[3] == 8 and 30 + h[9] + h[10] == 74
        assert h[7] == COMPRESSED and h[8] == UNCOMPRESSED and h[6] == expected_crc
    from download_model import ranged_download
    compressed = folder / "glove50.deflate"
    chunks = folder / "glove50.deflate.chunks"
    chunks.mkdir(exist_ok=True)
    for old in folder.glob("range-*.part"):
        target_part = chunks / (f"{int(old.stem.split('-')[1]):04d}.part")
        if not target_part.exists():
            target_part.write_bytes(old.read_bytes())
    ranged_download(URL, compressed, COMPRESSED, source_offset=OFFSET + 74, chunk_size=4 * 1024 * 1024)
    paths = [compressed]
    decoder, crc, length = zlib.decompressobj(-15), 0, 0
    target = folder / "glove.6B.50d.txt"
    with target.with_suffix(".txt.part").open("wb") as out:
        for path in paths:
            block = decoder.decompress(path.read_bytes())
            out.write(block)
            crc, length = zlib.crc32(block, crc), length + len(block)
    assert decoder.eof and length == UNCOMPRESSED and crc == expected_crc
    target.with_suffix(".txt.part").replace(target)
    print(target, flush=True)
