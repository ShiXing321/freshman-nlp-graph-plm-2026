"""Resume only when both completed results and required saved weights exist."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def same_value(actual, expected):
    if isinstance(actual, bool):
        return actual == expected
    if isinstance(actual, (int, float)):
        try:
            return float(actual) == float(expected)
        except (TypeError, ValueError):
            return False
    return str(actual) == str(expected)


def completed_artifacts(command):
    script = command[0].replace("\\", "/")
    options = {}
    positionals = []
    i = 1
    while i < len(command):
        value = command[i]
        if value.startswith("--"):
            key = value[2:].replace("-", "_")
            if i + 1 < len(command) and not command[i + 1].startswith("--"):
                options[key] = command[i + 1]
                i += 2
            else:
                options[key] = True
                i += 1
        else:
            positionals.append(value)
            i += 1
    if options.pop("no_shuffle", False):
        options["shuffle"] = False
    if positionals:
        options["task"] = positionals[0]
    if script.startswith("graph_beginner/task"):
        folder = ROOT / "/".join(script.split("/")[:2])
    elif script.endswith("train_numpy.py"):
        folder = ROOT / "nlp-beginner/legacy/task1"
    elif script.endswith("train_torch_bow.py"):
        folder = ROOT / "nlp-beginner/legacy/task2"
    elif script.endswith("train_torch.py"):
        task = {"classification": 2, "matching": 3, "ner": 4, "lm": 5}[options["task"]]
        folder = ROOT / f"nlp-beginner/legacy/task{task}"
    else:
        folder = ROOT / "transformers_tasks/local_runs"
    for path in (folder / "results").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        c = row.get("config", {})
        if row.get("status") != "completed" or not all(same_value(c.get(k), v) for k, v in options.items()):
            continue
        if script.startswith("graph_beginner/") or script.endswith("train_torch.py") or script.endswith("train_torch_bow.py"):
            required = [folder / "checkpoints" / (row["name"] + ".pt")]
        elif script.endswith("train_numpy.py"):
            required = [folder / "checkpoints" / (row["name"] + ".npz"), folder / "checkpoints" / (row["name"] + "_vectorizer.joblib")]
        elif script.endswith("bert_tasks.py"):
            required = [folder / "checkpoints" / c["task"] / "model.pt"]
        elif script.endswith("span_task.py"):
            key = ("uie_base_" if c.get("pretrained_uie") else "uie_") + c["augment"]
            required = [folder / "checkpoints" / key / "model.pt"]
        elif script.endswith("reward_task.py"):
            required = [folder / "checkpoints/reward/model.pt"]
        elif script.endswith("seq2seq_tasks.py"):
            checkpoint = folder / "checkpoints" / ("t5_" + c["task"] + c.get("tag", ""))
            required = [checkpoint / name for name in ["model.safetensors", "config.json", "tokenizer_config.json", "vocab.txt"]]
        elif script.endswith("ppo_task.py"):
            checkpoint = folder / "checkpoints" / ("ppo" + c.get("tag", ""))
            required = [checkpoint / name for name in ["adapter_model.safetensors", "adapter_config.json", "value_head.pt"]]
        elif script.endswith("llm_tasks.py"):
            if c["task"] in ["zero-shot", "mbti"]:
                required = [ROOT / ".cache/models/qwen05/config.json"]
            elif c["task"] in ["sft", "cpt"]:
                required = [folder / "checkpoints" / c["task"] / name for name in ["adapter_model.safetensors", "adapter_config.json"]]
            else:
                required = [folder / "checkpoints" / c["task"] / name for name in ["model.safetensors", "config.json", "tokenizer.json"]]
        else:
            continue
        if all(p.is_file() and p.stat().st_size for p in required):
            return True
    return False
