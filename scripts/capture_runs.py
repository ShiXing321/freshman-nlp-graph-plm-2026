"""Capture the browser page that directly executes a saved-model demonstration."""
import argparse
import hashlib
import html
import json
import os
import subprocess
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "学习报告/运行截图"
TASKS = [f"nlp{i}" for i in range(1, 6)] + [f"graph{i}" for i in range(1, 5)] + ["plm-bert", "plm-uie", "plm-reward", "plm-qa", "plm-filling", "plm-sft", "plm-cpt", "plm-ppo", "plm-scratch", "plm-scratch-sft", "plm-zero-shot", "plm-mbti"]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        task = urlparse(self.path).path.removeprefix("/")
        if task not in TASKS:
            self.send_error(404)
            return
        command = [sys.executable, "-X", "utf8", "-u", "scripts/demo_saved.py", task]
        started = datetime.now().astimezone().isoformat()
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1"}, timeout=600)
        stdout = proc.stdout + proc.stderr
        (FOLDER / (task + ".txt")).write_text(stdout, encoding="utf-8")
        record = {"task": task, "started": started, "finished": datetime.now().astimezone().isoformat(),
                  "command": ["python", "-X", "utf8", "-u", "scripts/demo_saved.py", task],
                  "exit_code": proc.returncode, "output_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
                  "capture": "Playwright screenshot of actual local execution page, no image editing"}
        (FOLDER / (task + ".json")).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        content = "<!doctype html><html lang='zh'><meta charset='utf-8'><title>运行输出</title><style>body{margin:0;background:#111827;color:#e5e7eb;font:16px/1.55 Consolas,'Microsoft YaHei',monospace}main{padding:26px 32px}.command{color:#93c5fd;border-bottom:1px solid #374151;padding-bottom:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:20px 0 0}</style><main><div class='command'>" + html.escape("python -X utf8 -u scripts/demo_saved.py " + task) + "</div><pre>" + html.escape(stdout) + "</pre></main></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(content.encode())

    def log_message(self, *args):
        pass


def main(tasks):
    FOLDER.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 8766), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="msedge", headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=1)
            for task in tasks:
                page.goto("http://127.0.0.1:8766/" + task, timeout=650000, wait_until="load")
                page.screenshot(path=str(FOLDER / (task + ".png")), full_page=True)
                record = json.loads((FOLDER / (task + ".json")).read_text(encoding="utf-8"))
                record["screenshot_sha256"] = hashlib.sha256((FOLDER / (task + ".png")).read_bytes()).hexdigest()
                (FOLDER / (task + ".json")).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                if record["exit_code"]:
                    raise RuntimeError(task + " failed; see original output")
                print(task, "captured", flush=True)
            browser.close()
    finally:
        server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="*")
    args = parser.parse_args()
    if any(task not in TASKS for task in args.tasks):
        parser.error("Unknown task")
    main(args.tasks or TASKS)
