"""Local tokenizer viewer and append-only rating form."""
import argparse
import json
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
TOKENIZERS = {}
PAGE = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>新生任务：本地工具</title>
<style>body{font:16px system-ui;max-width:900px;margin:40px auto;padding:0 20px;color:#243348;background:#f7f9fc}section{background:white;border:1px solid #dce4ee;border-radius:12px;padding:24px;margin:24px 0}textarea{width:95%;height:110px;padding:10px}button,select{padding:8px;margin:8px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf3fa;padding:16px}.tokens span{display:inline-block;border:1px solid #ccd7e5;padding:6px;margin:3px;border-radius:5px}h1{font-size:26px}</style>
<h1>新生任务：本地工具</h1><section><h2>分词查看</h2><p>输入一句话，观察 token、编号、偏移和解码结果。</p>
<select id="model"><option value="bert">中文小 BERT</option><option value="qwen">Qwen 0.5B</option><option value="scratch">自己训练的 BPE</option></select>
<textarea id="text">我在学习自然语言处理。Hello, NLP!</textarea><br><button id="encode">查看分词</button><div id="tokens" class="tokens"></div><pre id="details"></pre></section>
<section><h2>人工评价 PPO 输出</h2><p>阅读生成内容后，从准确性、表达和相关性三个方面给出评分。</p><button id="load">读取实际生成样例</button><select id="sample"></select><pre id="generation"></pre>
<label>质量评分 <select id="score"><option value="">请选择</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select>（1 差，5 好）</label>
<textarea id="comment" placeholder="可选：指出事实错误、语气或表达问题"></textarea><button id="save">保存本人评分</button><p id="message"></p></section>
<script>let samples=[];const el=id=>document.getElementById(id);el('encode').onclick=async()=>{let r=await fetch('/api/tokenize?model='+el('model').value+'&text='+encodeURIComponent(el('text').value));let d=await r.json();el('tokens').replaceChildren();(d.tokens||[]).forEach((t,i)=>{let s=document.createElement('span');s.textContent=t;s.title='id='+d.ids[i];el('tokens').append(s)});el('details').textContent=JSON.stringify(d,null,2)};
function show(){let s=samples[Number(el('sample').value)];el('generation').textContent=JSON.stringify(s,null,2);el('score').value=''}
el('load').onclick=async()=>{let r=await fetch('/api/samples');let d=await r.json();samples=d.samples||[];el('sample').replaceChildren();samples.forEach((s,i)=>{let o=document.createElement('option');o.value=i;o.textContent=(i+1)+' / 轮次 '+s.iteration;el('sample').append(o)});show();el('message').textContent=d.error||''};el('sample').onchange=show;
el('save').onclick=async()=>{if(!el('score').value||!samples.length){el('message').textContent='请先读取样例并选择评分。';return}let r=await fetch('/api/rating',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({index:Number(el('sample').value),score:Number(el('score').value),comment:el('comment').value})});el('message').textContent=(await r.json()).message||'保存失败'};</script></html>"""


def ppo_result_path():
    expanded = BASE / "local_runs/results/qwen_ppo_v2_expanded.json"
    if expanded.exists() and json.loads(expanded.read_text(encoding="utf-8")).get("status") == "completed":
        return expanded
    return BASE / "local_runs/results/qwen_ppo_v2.json"


class Handler(BaseHTTPRequestHandler):
    def reply(self, value, status=200):
        content = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        request = urlparse(self.path)
        if request.path == "/":
            content = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        elif request.path == "/api/samples":
            path = ppo_result_path()
            self.reply(json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"samples": [], "error": "PPO 结果尚未生成"})
        elif request.path == "/api/tokenize":
            args = parse_qs(request.query)
            name, text = args.get("model", ["bert"])[0], args.get("text", [""])[0][:10000]
            folders = {"bert": ROOT / ".cache/models/bert-small-zh", "qwen": ROOT / ".cache/models/qwen05", "scratch": BASE / "local_runs/checkpoints/scratch"}
            try:
                if name not in TOKENIZERS:
                    from transformers import AutoTokenizer
                    TOKENIZERS[name] = AutoTokenizer.from_pretrained(folders[name], local_files_only=True)
                tok = TOKENIZERS[name]
                encoded = tok(text, return_offsets_mapping=True)
                self.reply({"ids": encoded["input_ids"], "tokens": tok.convert_ids_to_tokens(encoded["input_ids"]), "offsets": encoded["offset_mapping"], "decoded": tok.decode(encoded["input_ids"], skip_special_tokens=True), "count": len(encoded["input_ids"])})
            except Exception as e:
                self.reply({"error": str(e)}, 400)
        else:
            self.reply({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/api/rating":
            return self.reply({"error": "not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > 20000:
                raise ValueError("评分过长")
            value = json.loads(self.rfile.read(length))
            index, score = int(value["index"]), int(value["score"])
            if score not in range(1, 6):
                raise ValueError("评分必须在 1—5 之间")
            result_path = ppo_result_path()
            result = json.loads(result_path.read_text(encoding="utf-8"))
            sample = result["samples"][index]
            if index < 0:
                raise ValueError("样例编号无效")
            path = BASE / "local_runs/human_ratings.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"time": datetime.now().astimezone().isoformat(), "result_file": result_path.name, "index": index, "score": score, "comment": value.get("comment", ""), "sample": sample, "source": "user entered in local UI"}, ensure_ascii=False) + "\n")
            self.reply({"message": "已保存到 human_ratings.jsonl"})
        except Exception as e:
            self.reply({"message": str(e)}, 400)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    print(f"本地工具 http://127.0.0.1:{args.port}，Ctrl+C 停止。", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
