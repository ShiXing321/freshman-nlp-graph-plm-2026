"""Download the CUDA wheel in resumable ranges, then install into the project venv."""
import concurrent.futures
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://s3.jcloud.sjtu.edu.cn/899a892efef34b1b944a19981040f55b-oss01/pytorch-wheels/cu124/torch-2.6.0%2Bcu124-cp311-cp311-win_amd64.whl"
SIZE = 2532350702
CHUNK = 64 * 1024 * 1024
folder = ROOT / ".cache/wheels"
folder.mkdir(parents=True, exist_ok=True)


def fetch(i):
    begin, end = i * CHUNK, min(SIZE, (i + 1) * CHUNK) - 1
    path = folder / f"torch-{i:03d}.part"
    if path.exists() and path.stat().st_size == end - begin + 1:
        return path
    subprocess.run(["curl.exe", "-sSfL", "--retry", "3", "--connect-timeout", "20", "--max-time", "1200",
                    "--range", f"{begin}-{end}", URL, "-o", str(path)], check=True)
    if path.stat().st_size != end - begin + 1:
        raise RuntimeError(f"range {i} has unexpected length {path.stat().st_size}")
    print(f"downloaded range {i}", flush=True)
    return path


if __name__ == "__main__":
    wheel = folder / "torch-2.6.0+cu124-cp311-cp311-win_amd64.whl"
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(fetch, range(math.ceil(SIZE / CHUNK))))
    with wheel.open("wb") as out:
        for path in paths:
            with path.open("rb") as f:
                while block := f.read(1024 * 1024):
                    out.write(block)
    import zipfile
    with zipfile.ZipFile(wheel) as z:
        corrupt = z.testzip()
        if corrupt:
            raise RuntimeError(f"wheel CRC failed: {corrupt}")
    subprocess.run([sys.executable, "-m", "pip", "install", str(wheel), "--index-url", "https://pypi.tuna.tsinghua.edu.cn/simple"], check=True)
