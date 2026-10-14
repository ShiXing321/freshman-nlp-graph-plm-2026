"""Held-out class diagnostics and a real ESIM attention plot."""
import json
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from functools import partial

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "nlp-beginner/legacy"
sys.path[:0] = [str(ROOT), str(BASE)]
from train_torch import read_rows, collate, tokenize
from models import ESIM, TextClassifier, BiLSTMCRF, packed_lstm


def main():
    torch.set_num_threads(4)
    out = ROOT / "运行记录/诊断"
    out.mkdir(parents=True, exist_ok=True)
    for task, kind, data_name in [("task2", "classification", "sentiment_kaggle"), ("task3", "matching", "snli")]:
        pattern = "cnn_glove*_kaggle.pt" if task == "task2" else "esim_glove*_expanded.pt"
        path = next((BASE / task / "checkpoints").glob(pattern))
        if task == "task2":
            choices = [p for p in (BASE / task / "checkpoints").glob("*_kaggle.pt") if "torch_bow" not in p.stem]
            path = max(choices, key=lambda p: json.loads((BASE / task / "results" / (p.stem + ".json")).read_text(encoding="utf-8"))["best_dev_score"])
        saved = torch.load(path, map_location="cpu", weights_only=True)
        c, vocab = saved["config"], saved["vocab"]
        model = TextClassifier(len(vocab), c["model"], c["dim"], c["hidden"]) if task == "task2" else ESIM(len(vocab), c["dim"], c["hidden"])
        model.load_state_dict(saved["state"])
        model.eval()
        rows = read_rows(data_name, 0, 42)["test"]
        truth, predicted, errors = [], [], []
        loader = DataLoader(rows, batch_size=64, collate_fn=partial(collate, task=kind, vocab=vocab, max_length=c["max_length"]))
        with torch.no_grad():
            for batch in loader:
                predicted.extend(model(*batch[:-1]).argmax(-1).tolist())
                truth.extend(batch[-1].tolist())
        for row, pred in zip(rows, predicted):
            if row["label"] != pred and len(errors) < 10:
                errors.append({"input": row, "prediction": pred})
        report = {"checkpoint": path.relative_to(ROOT).as_posix(), "test_examples": len(rows),
                  "classification_report": classification_report(truth, predicted, output_dict=True, zero_division=0),
                  "confusion_matrix": confusion_matrix(truth, predicted).tolist(), "errors": errors}
        (out / (task + "_classes.json")).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if task == "task3":
            row = next(r for r in rows if 5 <= len(tokenize(r["premise"])) <= 16 and 5 <= len(tokenize(r["hypothesis"])) <= 16)
            a, b, _ = collate([row], kind, vocab, c["max_length"])
            with torch.no_grad():
                ah = packed_lstm(model.encoder, model.embedding(a), a.ne(0))
                bh = packed_lstm(model.encoder, model.embedding(b), b.ne(0))
                attention = torch.bmm(ah, bh.transpose(1, 2)).softmax(-1)[0].numpy()
            fig, axis = plt.subplots(figsize=(9, 6))
            chart = axis.imshow(attention, cmap="Blues", aspect="auto", vmin=0)
            axis.set_xticks(range(b.shape[1]), tokenize(row["hypothesis"]), rotation=45, ha="right")
            axis.set_yticks(range(a.shape[1]), tokenize(row["premise"]))
            axis.set_xlabel("Hypothesis tokens")
            axis.set_ylabel("Premise tokens")
            axis.set_title("ESIM token attention on a held-out SNLI pair")
            fig.colorbar(chart, ax=axis, label="Attention probability")
            fig.tight_layout()
            fig.savefig(BASE / "task3/attention_example.png", dpi=150)
            plt.close(fig)
            (out / "esim_attention.json").write_text(json.dumps({"sample": row, "matrix": attention.tolist(), "row_sums": attention.sum(-1).tolist()}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(task, len(rows), "held-out examples", flush=True)
    from train_torch import TAGS
    from seqeval.metrics import classification_report as entity_report
    paths = list((BASE / "task4/checkpoints").glob("bilstm-crf_glove*.pt"))
    path = max(paths, key=lambda p: json.loads((BASE / "task4/results" / (p.stem + ".json")).read_text(encoding="utf-8"))["best_dev_score"])
    saved = torch.load(path, map_location="cpu", weights_only=True)
    c, vocab = saved["config"], saved["vocab"]
    model = BiLSTMCRF(len(vocab), dim=c["dim"], hidden=c["hidden"], use_case=c.get("use_case", False)).eval()
    model.load_state_dict(saved["state"])
    rows = read_rows("ner", 0, 42)["test"]
    truth, prediction = [], []
    loader = DataLoader(rows, batch_size=32, collate_fn=partial(collate, task="ner", vocab=vocab, max_length=c["max_length"], use_case=c.get("use_case", False)))
    with torch.no_grad():
        for batch in loader:
            ids, gold = batch[0], batch[-1]
            paths = model.crf.decode(model(*batch[:-1]), ids.ne(0))
            for i, tags in enumerate(paths):
                truth.append([TAGS[int(value)] for value in gold[i, :len(tags)]])
                prediction.append([TAGS[value] for value in tags])
    errors = [{"tokens": row["tokens"], "gold": gold, "prediction": pred} for row, gold, pred in zip(rows, truth, prediction) if gold != pred][:5]
    data = {"checkpoint": path.relative_to(ROOT).as_posix(), "test_examples": len(rows),
            "entity_report": entity_report(truth, prediction, output_dict=True, zero_division=0), "errors": errors}
    (out / "task4_entities.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else value), encoding="utf-8")
    print("task4", len(rows), "held-out examples", flush=True)


if __name__ == "__main__":
    main()
