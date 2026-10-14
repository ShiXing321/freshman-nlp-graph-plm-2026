"""Link prediction; held-out positives never enter training message-passing edges."""
import argparse
import copy
import sys
import time
from pathlib import Path
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.utils import negative_sampling, to_undirected

BASE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BASE), str(BASE.parent), str(BASE / "task1/code")]
from common.experiment import Run, device, seed_all
from models import GNN
from train import dataset


def decode(z, edge):
    return (z[edge[0]] * z[edge[1]]).sum(-1)


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    scores = decode(model(data.x, data.edge_index), data.edge_label_index).cpu().numpy()
    y = data.edge_label.cpu().numpy()
    return {"auc": roc_auc_score(y, scores), "average_precision": average_precision_score(y, scores)}


def main(args):
    seed_all(args.seed)
    ds = dataset(args.dataset)
    original = ds[0]
    all_edges = original.edge_index.clone()
    split = RandomLinkSplit(num_val=.1, num_test=.1, is_undirected=True, add_negative_train_samples=True, neg_sampling_ratio=1., disjoint_train_ratio=.2)
    train, val, test = split(original)
    # Static negatives sampled by RandomLinkSplit exclude positives across all splits.
    dev = torch.device(args.device) if args.device else device()
    model = GNN(ds.num_features, args.hidden, args.model, args.hidden, args.layers).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    run = Run(BASE / "task2/results", f"{args.dataset}_{args.model}_{args.mode}_l{args.layers}_lr{args.lr}" + args.tag, vars(args) | {"nodes": original.num_nodes, "edges": original.num_edges, "split": "10% val/10% test; 20% disjoint supervision", "actual_device": str(dev), "sampler": "RandomNodeLoader induced partitions" if args.mode == "sampled" else None})
    best, state, seconds = -1., None, 0.
    # Sample induced subgraphs using the framework. Attach global IDs to recover held-out labels.
    sampled_data = train.clone()
    sampled_data.n_id = torch.arange(train.num_nodes)
    del sampled_data.edge_label_index
    del sampled_data.edge_label
    loader = RandomNodeLoader(sampled_data, num_parts=args.parts, shuffle=True, num_workers=0) if args.mode == "sampled" else [train]
    val, test = val.to(dev), test.to(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        model.train()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        total, count = 0., 0
        for batch in loader:
            if args.mode == "sampled":
                mapping = torch.full((original.num_nodes,), -1, dtype=torch.long)
                mapping[batch.n_id] = torch.arange(batch.num_nodes)
                local = mapping[train.edge_label_index]
                keep = (local >= 0).all(0)
                edges, labels = local[:, keep], train.edge_label[keep]
            else:
                edges, labels = batch.edge_label_index, batch.edge_label
            if labels.numel() == 0:
                continue
            batch = batch.to(dev)
            edges, labels = edges.to(dev), labels.to(dev)
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(decode(model(batch.x, batch.edge_index), edges), labels)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * labels.numel()
            count += labels.numel()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        seconds += elapsed
        metrics = evaluate(model, val)
        run.epoch(epoch=epoch + 1, loss=total / max(count, 1), supervised_edges=count, train_seconds=elapsed, dev=metrics)
        if metrics["auc"] > best:
            best, state = metrics["auc"], copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    metrics = evaluate(model, test)
    ckpt = BASE / "task2/checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.cpu().state_dict(), "config": vars(args)}, ckpt / (run.data["name"] + ".pt"))
    run.finish(best_dev_auc=best, test=metrics, train_seconds=seconds,
               peak_cuda_mb=torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["Cora", "CiteSeer", "Flickr"], default="Cora")
    p.add_argument("--model", choices=["gcn", "gat", "sage", "gin"], default="gcn")
    p.add_argument("--mode", choices=["full", "sampled"], default="full")
    p.add_argument("--parts", type=int, default=4)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="")
    p.add_argument("--tag", default="")
    main(p.parse_args())
