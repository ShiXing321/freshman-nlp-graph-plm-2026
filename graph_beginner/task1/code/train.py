"""Node classification with full-graph or PyG RandomNodeLoader training."""
import argparse
import copy
import sys
import time
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.datasets import Planetoid, Flickr
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.transforms import NormalizeFeatures

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE.parent
sys.path[:0] = [str(BASE), str(ROOT)]
from common.experiment import Run, device, seed_all
from models import GNN


def dataset(name):
    folder = BASE / "task1/data/raw" / name
    return (Flickr(str(folder)) if name == "Flickr" else Planetoid(str(folder), name, transform=NormalizeFeatures()))


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    pred = model(data.x, data.edge_index)[mask].argmax(-1).cpu().numpy()
    truth = data.y[mask].cpu().numpy()
    return {"accuracy": accuracy_score(truth, pred), "macro_f1": f1_score(truth, pred, average="macro", zero_division=0)}


def main(args):
    seed_all(args.seed)
    ds = dataset(args.dataset)
    data = ds[0]
    dev = torch.device(args.device) if args.device else device()
    model = GNN(ds.num_features, ds.num_classes, args.model, args.hidden, args.layers).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    loader = RandomNodeLoader(data, num_parts=args.parts, shuffle=True, num_workers=0) if args.mode == "sampled" else [data]
    full = data.clone().to(dev)
    config = vars(args) | {"nodes": data.num_nodes, "edges": data.num_edges, "actual_device": str(dev), "sampler": "RandomNodeLoader induced partitions" if args.mode == "sampled" else None}
    run = Run(BASE / "task1/results", f"{args.dataset}_{args.model}_{args.mode}_l{args.layers}_lr{args.lr}" + args.tag, config)
    best, state, train_seconds = -1., None, 0.
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        total, count = 0., 0
        model.train()
        for batch in loader:
            batch = batch.to(dev)
            mask = batch.train_mask
            if not mask.any():
                continue
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(batch.x, batch.edge_index)[mask], batch.y[mask])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * int(mask.sum())
            count += int(mask.sum())
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        train_seconds += elapsed
        metrics = evaluate(model, full, full.val_mask)
        run.epoch(epoch=epoch + 1, loss=total / max(count, 1), train_seconds=elapsed, dev=metrics)
        if metrics["accuracy"] > best:
            best, state = metrics["accuracy"], copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    ckpt = BASE / "task1/checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.cpu().state_dict(), "config": config}, ckpt / (run.data["name"] + ".pt"))
    model.to(dev)
    run.finish(best_dev_accuracy=best, test=evaluate(model, full, full.test_mask), train_seconds=train_seconds,
               peak_cuda_mb=torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["Cora", "CiteSeer", "Flickr"], default="Cora")
    p.add_argument("--model", choices=["gcn", "gat", "sage", "gin"], default="gcn")
    p.add_argument("--mode", choices=["full", "sampled"], default="full")
    p.add_argument("--parts", type=int, default=4)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="")
    p.add_argument("--tag", default="")
    main(p.parse_args())
