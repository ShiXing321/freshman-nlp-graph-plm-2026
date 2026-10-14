"""Metric comparisons drawn directly from the selected experiment records."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compare(records, folder, domain, task):
    target = Path(folder) / "metric_comparison.png"
    colors = ["#345d8b", "#3a9b8e"]
    if domain == "nlp" and task in ["task1", "task2"]:
        labels, accuracy, f1 = [], [], []
        for _, row in records:
            c = row["config"]
            if task == "task1":
                if c["batch"] == 0:
                    label = "Full batch"
                elif c["batch"] == 1:
                    label = "SGD"
                elif not c["shuffle"]:
                    label = "No shuffle"
                elif c["loss"] == "logistic":
                    label = "OVR logistic"
                elif c["lr"] == .05:
                    label = "TF-IDF lr=.05"
                else:
                    label = {"count": "Word counts", "binary": "Binary", "tfidf": "TF-IDF"}[c["features"]] + (" + bigram" if c["ngram"] == 2 else "")
            elif "model" not in c:
                label = "TF-IDF " + c["loss"]
            else:
                label = {"cnn": "CNN", "rnn": "BiLSTM", "softmax": "Mean embeddings"}[c["model"]]
                if c["model"] != "softmax":
                    label += " GloVe" if c.get("glove") else " random"
            metrics = row.get("test", {})
            labels.append(label)
            accuracy.append(row.get("test_accuracy", metrics.get("accuracy")))
            f1.append(row.get("test_macro_f1", metrics.get("macro_f1")))
        figure, axis = plt.subplots(figsize=(12, 4.8))
        x = np.arange(len(labels))
        axis.bar(x - .18, accuracy, .36, label="Accuracy", color=colors[0])
        axis.bar(x + .18, f1, .36, label="Macro-F1", color=colors[1])
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.set_ylim(0, 1)
        axis.set_ylabel("Held-out test metric")
        axis.legend(frameon=False)
        axis.grid(axis="y", alpha=.2)
    elif domain == "graph" and task in ["task1", "task2"]:
        figure, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        for axis, dataset in zip(axes, ["Cora", "CiteSeer", "Flickr"]):
            for shift, mode, color in [(-.18, "full", colors[0]), (.18, "sampled", colors[1])]:
                scores = []
                for model in ["gcn", "gat", "sage", "gin"]:
                    rows = [r for _, r in records if r["config"].get("dataset") == dataset and r["config"].get("model") == model and r["config"].get("mode") == mode and r["config"].get("layers") == 2 and r["config"].get("lr") == .01 and "_seed" not in r["name"] and "_ablation" not in r["name"]]
                    scores.append(rows[0]["test"]["accuracy" if task == "task1" else "auc"] if rows else float("nan"))
                axis.bar(np.arange(4) + shift, scores, .36, color=color, label=mode)
            axis.set_xticks(np.arange(4), ["GCN", "GAT", "SAGE", "GIN"])
            axis.set_title(dataset)
            axis.set_ylim(0, 1)
            axis.grid(axis="y", alpha=.2)
        axes[0].set_ylabel("Test accuracy" if task == "task1" else "Test ROC-AUC")
        axes[-1].legend(frameon=False)
    elif domain == "graph" and task == "task3":
        figure, axes = plt.subplots(2, 1, figsize=(12, 7))
        keys = [(model, pool) for model in ["gcn", "gat", "sage", "gin"] for pool in ["mean", "max", "min"]]
        for axis, dataset, metric in zip(axes, ["MUTAG", "ZINC"], ["accuracy", "mae"]):
            for shift, mode, color in [(-.18, "full", colors[0]), (.18, "batch", colors[1])]:
                scores = []
                for model, pool in keys:
                    rows = [r for _, r in records if r["config"].get("dataset") == dataset and r["config"].get("model") == model and r["config"].get("pool") == pool and r["config"].get("mode") == mode and "_ablation" not in r["name"]]
                    scores.append(rows[0]["test"][metric] if rows else float("nan"))
                axis.bar(np.arange(len(keys)) + shift, scores, .36, color=color, label=mode)
            axis.set_xticks(np.arange(len(keys)), [m.upper() + " / " + p for m, p in keys], rotation=35, ha="right")
            axis.set_title(dataset + (" classification" if dataset == "MUTAG" else " property regression"))
            axis.set_ylabel("Test " + metric)
            axis.legend(frameon=False)
            axis.grid(axis="y", alpha=.2)
    elif domain == "graph" and task == "task4":
        figure, axis = plt.subplots(figsize=(9, 4))
        metrics = ["mrr", "hits@1", "hits@3", "hits@10"]
        for i, (_, row) in enumerate(records):
            axis.bar(np.arange(4) + (i - 1) * .23, [row["test"][key] for key in metrics], .23, label=row["config"]["model"].upper())
        axis.set_xticks(np.arange(4), ["MRR", "Hits@1", "Hits@3", "Hits@10"])
        axis.set_ylabel("Filtered head/tail test metric")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=.2)
        axis.legend(frameon=False)
    else:
        return None
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return target
