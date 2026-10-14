"""TU graph classification and ZINC graph regression; mean/max/min pooling."""
import argparse
import copy
import sys
import time
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.datasets import TUDataset, ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

BASE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BASE), str(BASE.parent)]
from common.experiment import Run, device, seed_all
from models import GraphPredictor


def main(args):
    seed_all(args.seed)
    regression = args.dataset == "ZINC"
    if regression:
        folder = BASE / "task3/data/raw/ZINC-HF"
        if list(folder.glob("*.parquet")):
            import pyarrow.parquet as pq
            subsets = {}
            for split, key in [("train", "train"), ("dev", "validation"), ("test", "test")]:
                path = next(folder.glob(key + "-*.parquet"))
                table = pq.read_table(path)
                # This mirror is the full ZINC split; take a fixed, documented subset.
                take = min(len(table), args.limit if split == "train" else min(args.limit, 1000)) if args.limit else len(table)
                rng = __import__("numpy").random.default_rng(args.seed)
                records = table.take(rng.choice(len(table), take, replace=False)).to_pylist()
                subsets[split] = [Data(x=torch.tensor(r["node_feat"], dtype=torch.long).reshape(-1, 1),
                                      edge_index=torch.tensor(r["edge_index"], dtype=torch.long),
                                      edge_attr=torch.tensor(r["edge_attr"], dtype=torch.long),
                                      y=torch.tensor(r["y"], dtype=torch.float).reshape(1), num_nodes=int(r["num_nodes"])) for r in records]
        else:
            subsets = {s: ZINC(str(BASE / "task3/data/raw/ZINC"), subset=True, split=key) for s, key in [("train", "train"), ("dev", "val"), ("test", "test")]}
        in_dim, out_dim, atom_types = 32, 1, 28
    else:
        ds = TUDataset(str(BASE / "task3/data/raw/MUTAG"), "MUTAG")
        idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed))
        n = len(ds)
        subsets = {"train": ds[idx[:int(n * .8)]], "dev": ds[idx[int(n * .8):int(n * .9)]], "test": ds[idx[int(n * .9):]]}
        in_dim, out_dim, atom_types = ds.num_features, ds.num_classes, 0
    if args.limit:
        subsets = {s: ds[:args.limit if s == "train" else min(args.limit, 1000)] for s, ds in subsets.items()}
    loaders = {s: DataLoader(ds, batch_size=args.batch if args.mode == "batch" else len(ds), shuffle=s == "train", num_workers=0) for s, ds in subsets.items()}
    dev = device()
    model = GraphPredictor(in_dim, out_dim, args.model, args.pool, args.hidden, args.layers, atom_types).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    run = Run(BASE / "task3/results", f"{args.dataset}_{args.model}_{args.pool}_{args.mode}_n{args.limit}_lr{args.lr}_l{args.layers}" + args.tag, vars(args) | {"problem": "regression" if regression else "classification", "counts": {s: len(ds) for s, ds in subsets.items()}, "bond_features": "available in data; these four baseline encoders use topology and atom IDs only", "data_source": "graphs-datasets/ZINC full split, seeded subset" if regression and list((BASE / "task3/data/raw/ZINC-HF").glob("*.parquet")) else "PyG dataset"})
    @torch.no_grad()
    def evaluate(split):
        model.eval()
        predictions, truth = [], []
        for batch in loaders[split]:
            batch = batch.to(dev)
            out = model(batch)
            predictions.extend((out.reshape(-1) if regression else out.argmax(-1)).cpu().tolist())
            truth.extend(batch.y.reshape(-1).cpu().tolist())
        if regression:
            return {"mae": sum(abs(x - y) for x, y in zip(predictions, truth)) / len(truth)}
        return {"accuracy": accuracy_score(truth, predictions), "macro_f1": f1_score(truth, predictions, average="macro", zero_division=0)}
    best, state, seconds = float("inf") if regression else -1, None, 0.
    for epoch in range(args.epochs):
        model.train()
        start = time.perf_counter()
        total, count = 0., 0
        for batch in loaders["train"]:
            batch = batch.to(dev)
            opt.zero_grad()
            out = model(batch)
            loss = torch.nn.functional.l1_loss(out.reshape(-1), batch.y.float().reshape(-1)) if regression else torch.nn.functional.cross_entropy(out, batch.y.reshape(-1))
            loss.backward()
            opt.step()
            total += float(loss.detach()) * batch.num_graphs
            count += batch.num_graphs
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        seconds += elapsed
        metrics = evaluate("dev")
        run.epoch(epoch=epoch + 1, loss=total / count, train_seconds=elapsed, dev=metrics)
        score = metrics["mae" if regression else "accuracy"]
        if (score < best) if regression else (score > best):
            best, state = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    ckpt = BASE / "task3/checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    metrics = evaluate("test")
    torch.save({"state": model.cpu().state_dict(), "config": vars(args)}, ckpt / (run.data["name"] + ".pt"))
    run.finish(best_dev_score=best, test=metrics, train_seconds=seconds)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["MUTAG", "ZINC"], default="MUTAG")
    p.add_argument("--model", choices=["gcn", "gat", "sage", "gin"], default="gcn")
    p.add_argument("--pool", choices=["mean", "max", "min"], default="mean")
    p.add_argument("--mode", choices=["full", "batch"], default="batch")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="")
    main(p.parse_args())
