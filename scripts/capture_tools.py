"""Capture the actual tokenizer viewer after a successful interactive request."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / "学习报告/运行截图"
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    page.goto("http://127.0.0.1:8765", wait_until="load")
    page.locator("#text").fill("我在学习自然语言处理。Hello, NLP!")
    page.locator("#encode").click()
    page.wait_for_function("document.getElementById('details').textContent.includes('decoded')")
    page.locator("section").first.screenshot(path=str(folder / "plm-tools.png"))
    output = page.locator("#details").inner_text()
    (folder / "plm-tools.txt").write_text(output, encoding="utf-8")
    (folder / "plm-tools.json").write_text(json.dumps({"started": datetime.now().astimezone().isoformat(),
               "url": "http://127.0.0.1:8765", "action": "input text, click tokenizer button, wait for actual API response",
               "exit_code": 0, "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
               "screenshot_sha256": hashlib.sha256((folder / "plm-tools.png").read_bytes()).hexdigest(),
               "capture": "Playwright element screenshot of original running tokenizer UI"}, ensure_ascii=False, indent=2), encoding="utf-8")
    browser.close()
print("Tokenizer viewer captured")
