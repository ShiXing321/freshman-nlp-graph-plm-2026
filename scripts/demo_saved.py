"""Load saved models and run held-out examples for the submission screenshots."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def show(value):
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


def nlp(task):
    base = ROOT / "nlp-beginner/legacy"
    sys.path.insert(0, str(base))
    from train_torch import read_rows, collate, evaluate, TAGS
    from models import TextClassifier, ESIM, BiLSTMCRF, CharLM
    from torch.utils.data import DataLoader
    from functools import partial
    paths = list((base / f"task{task}/checkpoints").glob("*.pt" if task > 1 else "*.npz"))
    if task < 3:
        preferred = [p for p in paths if "kaggle" in p.stem]
    elif task in [3, 4]:
        preferred = [p for p in paths if "expanded" in p.stem]
    else:
        preferred = paths
    paths = preferred or paths
    if task == 1:
        import joblib
        from sklearn.metrics import accuracy_score, f1_score
        path = next((p for p in paths if "tfidf_n2" in p.stem), paths[0])
        weights = np.load(path)
        vec = joblib.load(path.with_name(path.stem + "_vectorizer.joblib"))
        data_name = "sentiment_kaggle" if "kaggle" in path.stem else "sentiment"
        rows = read_rows(data_name, 0, 42)["test"]
        prediction = (vec.transform([r["text"] for r in rows]) @ weights["weights"] + weights["bias"]).argmax(1)
        show({"checkpoint": path.relative_to(ROOT).as_posix(), "test_examples": len(rows),
              "accuracy": accuracy_score([r["label"] for r in rows], prediction),
              "macro_f1": f1_score([r["label"] for r in rows], prediction, average="macro")})
        for row, pred in list(zip(rows, prediction))[:5]:
            show({"text": row["text"], "gold": row["label"], "prediction": int(pred)})
        return
    paths = [p for p in paths if "torch_bow" not in p.stem]
    path = next((p for p in paths if "glove" in p.stem), paths[0])
    if task == 2:
        path = max(paths, key=lambda p: json.loads((base / "task2/results" / (p.stem + ".json")).read_text(encoding="utf-8"))["best_dev_score"])
    if task == 4:
        paths = list((base / "task4/checkpoints").glob("*.pt"))
        path = max(paths, key=lambda p: json.loads((base / "task4/results" / (p.stem + ".json")).read_text(encoding="utf-8"))["best_dev_score"])
    snapshot = torch.load(path, map_location="cpu", weights_only=True)
    c, vocab = snapshot["config"], snapshot["vocab"]
    if task == 5:
        model = CharLM(len(vocab), c["model"], c["dim"], c["hidden"]).eval()
        model.load_state_dict(snapshot["state"])
        reverse = {v: k for k, v in vocab.items()}
        ids = torch.tensor([[vocab.get(x, 0) for x in "春风"]])
        state, text = None, "春风"
        with torch.no_grad():
            for _ in range(80):
                logits, state = model(ids, state)
                ids = torch.multinomial((logits[:, -1] / .7).softmax(-1), 1)
                text += reverse[int(ids.item())]
        show({"checkpoint": path.relative_to(ROOT).as_posix(), "temperature": .7, "generated": text})
        return
    kind = {2: "classification", 3: "matching", 4: "ner"}[task]
    model = TextClassifier(len(vocab), c["model"], c["dim"], c["hidden"]) if task == 2 else ESIM(len(vocab), c["dim"], c["hidden"]) if task == 3 else BiLSTMCRF(len(vocab), dim=c["dim"], hidden=c["hidden"], use_case=c.get("use_case", False))
    model.load_state_dict(snapshot["state"])
    model.eval()
    data_name = c.get("data_name", "sentiment") if task == 2 else "snli" if task == 3 else "ner"
    rows = read_rows(data_name, 0, 42)["test"][:128]
    loader = DataLoader(rows, batch_size=32, collate_fn=partial(collate, task=kind, vocab=vocab, max_length=c["max_length"], use_case=c.get("use_case", False)))
    show({"checkpoint": path.relative_to(ROOT).as_posix(), "test_examples": len(rows), "metrics": evaluate(model, loader, kind, torch.device("cpu"))})
    row = rows[0]
    batch = collate([row], kind, vocab, c["max_length"], use_case=c.get("use_case", False))
    with torch.no_grad():
        output = model(*batch[:-1])
        pred = [TAGS[i] for i in model.crf.decode(output, batch[0].ne(0))[0]] if task == 4 else int(output.argmax(-1).item())
    show({"input": row, "prediction": pred})


def graph(task):
    base = ROOT / "graph_beginner"
    sys.path[:0] = [str(base), str(base / "task1/code")]
    from common.experiment import seed_all
    from models import GNN, GraphPredictor
    if task in [1, 2]:
        node = module("graph_beginner/task1/code/train.py", "node_demo")
        ds = node.dataset("Cora")
        for kind in ["gcn", "gat", "sage", "gin"]:
            seed_all(42)
            data = None
            if task == 2:
                from torch_geometric.transforms import RandomLinkSplit
                # Match training: split edges before constructing the random model.
                _, _, data = RandomLinkSplit(num_val=.1, num_test=.1, is_undirected=True, add_negative_train_samples=True, neg_sampling_ratio=1., disjoint_train_ratio=.2)(ds[0])
            folder = base / f"task{task}/checkpoints"
            choices = sorted(folder.glob(f"Cora_{kind}_full_l2_lr0.01*.pt"))
            path = next((p for p in choices if "isolated" in p.stem), choices[0])
            snapshot = torch.load(path, map_location="cpu", weights_only=True)
            c = snapshot["config"]
            model = GNN(ds.num_features, ds.num_classes if task == 1 else c["hidden"], kind, c["hidden"], c["layers"]).eval()
            model.load_state_dict(snapshot["state"])
            if task == 1:
                metric = node.evaluate(model, ds[0], ds[0].test_mask)
            else:
                from sklearn.metrics import roc_auc_score, average_precision_score
                with torch.no_grad():
                    z = model(data.x, data.edge_index)
                    scores = (z[data.edge_label_index[0]] * z[data.edge_label_index[1]]).sum(-1).numpy()
                metric = {"auc": roc_auc_score(data.edge_label.numpy(), scores), "average_precision": average_precision_score(data.edge_label.numpy(), scores)}
            record = json.loads((base / f"task{task}/results" / (path.stem + ".json")).read_text(encoding="utf-8"))
            for key, value in metric.items():
                tolerance = 1e-5 if task == 2 and key == "average_precision" else 1e-6
                assert abs(value - record["test"][key]) < tolerance, (kind, key, value, record["test"][key])
            show({"checkpoint": path.relative_to(ROOT).as_posix(), "dataset": "Cora", "model": kind, "test": metric})
    elif task == 3:
        from torch_geometric.datasets import TUDataset
        from torch_geometric.loader import DataLoader
        ds = TUDataset(str(base / "task3/data/raw/MUTAG"), "MUTAG")
        indices = torch.randperm(len(ds), generator=torch.Generator().manual_seed(42))
        test = ds[indices[int(len(ds) * .9):]]
        for pool in ["mean", "max", "min"]:
            path = next((base / "task3/checkpoints").glob(f"MUTAG_gin_{pool}_batch_n0_lr0.001_l2.pt"))
            snapshot = torch.load(path, map_location="cpu", weights_only=True)
            c = snapshot["config"]
            model = GraphPredictor(ds.num_features, ds.num_classes, "gin", pool, c["hidden"], c["layers"]).eval()
            model.load_state_dict(snapshot["state"])
            batch = next(iter(DataLoader(test, batch_size=len(test))))
            with torch.no_grad():
                prediction = model(batch).argmax(-1)
            show({"pooling": pool, "test_graphs": len(test), "accuracy": float((prediction == batch.y).float().mean()), "gold": batch.y.tolist(), "prediction": prediction.tolist()})
    else:
        sys.path.insert(0, str(base / "task4/code"))
        sys.modules.pop("models", None)
        kg = module("graph_beginner/task4/code/train.py", "kg_demo")
        triples, _, _ = kg.load_data()
        known = set(map(tuple, torch.cat(list(triples.values())).tolist()))
        for kind in ["transe", "rotate", "conve"]:
            paths = list((base / "task4/checkpoints").glob(kind + "*.pt"))
            path = next((p for p in paths if "full" in p.stem), paths[0])
            snap = torch.load(path, map_location="cpu", weights_only=True)
            model = {"transe": kg.TransE, "rotate": kg.RotatE, "conve": kg.ConvE}[kind](len(snap["entities"]), len(snap["relations"]), snap["config"]["dim"])
            model.load_state_dict(snap["state"])
            metric = kg.evaluate(model, triples["test"], len(snap["entities"]), known, 50, torch.device("cpu"))
            show({"model": kind, "test": metric})


def plm(task):
    folder = ROOT / "transformers_tasks/local_runs"
    sys.path.insert(0, str(folder))
    if task == "bert":
        from bert_tasks import prepare_model, BertTask, load_rows, evaluate
        from transformers import logging
        logging.set_verbosity_error()
        pretrained, tok = prepare_model()
        for kind in ["classification", "pointwise", "dssm", "sbert", "simcse", "pet", "ptuning"]:
            snap = torch.load(folder / f"checkpoints/{kind}/model.pt", map_location="cpu", weights_only=False)
            rows, _ = load_rows(kind, 42)
            model = BertTask(pretrained, tok, kind, snap["config"]["labels"])
            model.load_state_dict(snap["state"])
            sample = rows["test"][:128]
            metric, _ = evaluate(model, sample, torch.device("cpu"), 16, snap.get("threshold"))
            show({"model": kind, "test_examples": len(sample), "metrics": metric})
    elif task == "uie":
        from span_task import prepare_span, SpanModel, data, evaluate
        base = (folder / "checkpoints/uie_base_none/model.pt").exists()
        pretrained, tok = prepare_span(base)
        for arm in (["none", "negative", "swap", "fill"] if base else ["none", "negative", "swap"]):
            path = folder / "checkpoints" / (("uie_base_" if base else "uie_") + arm) / "model.pt"
            if not path.exists():
                continue
            snap = torch.load(path, map_location="cpu", weights_only=True)
            model = SpanModel(pretrained)
            model.load_state_dict(snap["state"])
            metric, samples = evaluate(model, tok, data(42)["test"], torch.device("cpu"), 8, snap["threshold"])
            show({"augmentation": arm, "test": metric, "example": samples[:1]})
    elif task == "reward":
        from bert_tasks import prepare_model
        from reward_task import RewardModel
        p, tok = prepare_model()
        model = RewardModel(p).eval()
        snap = torch.load(folder / "checkpoints/reward/model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(snap["state"])
        sentences = ["这家店的味道很好，下次还会再来。", "这家店的食物坏了，服务也很差。"]
        with torch.no_grad():
            scores = model(tok(sentences, padding=True, return_tensors="pt")).tolist()
        show(dict(zip(sentences, scores)))
    elif task in ["zero-shot", "mbti"]:
        from llm_tasks import qwen, chat
        model, tok = qwen()
        if task == "mbti":
            questions = json.loads((ROOT / "transformers_tasks/LLM/llms_mbti/mbti_questions.json").read_text(encoding="utf-8"))
            number, question = next(iter(questions.items()))
            prompt = "请选择A或B，只输出一个字母。\n" + question["question"].replace("\\n", "\n")
            show({"question": number, "prompt": prompt, "output": chat(model, tok, prompt, 8)})
        else:
            schema = json.loads((folder / "results/qwen_zero_shot_v2_expanded.json").read_text(encoding="utf-8"))["config"]["category_schema"]
            prompts = ["将评论分类。只回答一个类别。类别：" + "、".join(schema.values()) + "。评论：苹果很新鲜，包装也很好。",
                       "判断两个句子表达的意思是否相同，只回答相同或不同。句子1：今天北京下雨了。句子2：北京今天有雨。",
                       "抽取原文中属于‘人物’的信息。只输出JSON字符串数组。原文：李白是唐代诗人。"]
            study_path = folder / "results/qwen_zero_shot_prompt_study.json"
            if study_path.exists():
                study = json.loads(study_path.read_text(encoding="utf-8"))
                if study.get("status") == "completed":
                    from zero_shot_prompt_study import prompt as extraction_prompt
                    prompts = prompts[:2]
                    rows = study["tasks"]["extraction"]["examples"]
                    for positive in [True, False]:
                        row = next(r for r in rows if bool(r["gold"]) == positive)
                        prompts.append(extraction_prompt(row, study["selected_template"]))
            for prompt in prompts:
                show({"prompt": prompt, "output": chat(model, tok, prompt, 80)})
    else:
        import subprocess
        kind = {"qa": "t5_qa", "filling": "t5_filling"}.get(task, task)
        text = "问题：李白是哪一朝的诗人？ 原文：李白是唐代诗人。" if task == "qa" else '"[MASK]，著名画家，毕业于中央美术学院"中[MASK]位置的文本是：' if task == "filling" else "请用一句话解释梯度下降。"
        command = [sys.executable, "-X", "utf8", str(folder / "infer_saved.py"), kind, "--text", text]
        expanded = folder / "checkpoints" / (kind + "_expanded")
        if task in ["qa", "filling", "ppo"] and expanded.exists():
            command += ["--checkpoint", str(expanded)]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if args.task.startswith("nlp"):
        nlp(int(args.task[-1]))
    elif args.task.startswith("graph"):
        graph(int(args.task[-1]))
    else:
        plm(args.task.removeprefix("plm-"))
