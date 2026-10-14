"""Validate dependencies and inference in the newly created isolated environment."""
import json
import os
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "运行记录/独立环境"


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    config = (Path(sys.executable).parents[1] / "pyvenv.cfg").read_text(encoding="utf-8")
    assert "include-system-site-packages = false" in config.lower()
    checks = []
    commands = [(["-m", "pip", "check"], "dependencies"),
                (["scripts/verify_algorithms.py"], "numpy-crf"),
                (["scripts/verify_plm_math.py"], "ppo-math"),
                (["scripts/verify_kge_ranking.py"], "kge-ranking"),
                (["scripts/demo_saved.py", "nlp1"], "numpy-saved-inference"),
                (["scripts/demo_saved.py", "graph1"], "graph-saved-inference"),
                (["transformers_tasks/local_runs/infer_saved.py", "sft", "--text", "请用一句话解释梯度下降。", "--output", str(FOLDER / "sft.json")], "qwen-saved-inference")]
    for command, name in commands:
        with (FOLDER / (name + ".txt")).open("w", encoding="utf-8") as log:
            proc = subprocess.run([sys.executable, "-X", "utf8", *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  env=os.environ | {"PYTHONUTF8": "1", "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"})
        checks.append({"name": name, "command": command, "passed": proc.returncode == 0, "exit_code": proc.returncode})
        print(name, proc.returncode, flush=True)
    result = {"status": "passed" if all(row["passed"] for row in checks) else "failed", "python": str(Path(sys.executable).relative_to(ROOT)),
              "include_system_site_packages": False, "checks": checks}
    (ROOT / "运行记录/独立环境核查.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result["status"] == "passed"


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
