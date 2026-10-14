"""FB15k-237 training and filtered head/tail ranking for TransE/RotatE/ConvE."""
import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
import torch
from torch.utils.data import DataLoader

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE.parent
sys.path.insert(0, str(ROOT))
from common.experiment import Run, download, device, seed_all
from models import TransE, RotatE, ConvE


def load_data():
    raw = BASE / "task4/data/raw/FB15k-237"
    rows = {}
    for split in ["train", "valid", "test"]:
        path = raw / f"{split}.txt"
        if not path.exists():
            api = download(f"https://api.github.com/repos/DeepGraphLearning/KnowledgeGraphEmbedding/contents/data/FB15k-237/{split}.txt", raw / f"{split}.api.json")
            import base64
            info = json.loads(api.read_text(encoding="utf-8"))
            if not info.get("content"):
                blob = download(info["git_url"], raw / f"{split}.blob.json", timeout=1200)
                info = json.loads(blob.read_text(encoding="utf-8"))
            path.write_bytes(base64.b64decode(info["content"]))
        rows[split] = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines() if line]
    # Entity/relation vocabularies are transductive metadata, not labels for training.
    entities = {e: i for i, e in enumerate(sorted({e for triples in rows.values() for h, r, t in triples for e in [h, t]}))}
    relations = {r: i for i, r in enumerate(sorted({r for triples in rows.values() for h, r, t in triples}))}
    tensors = {s: torch.tensor([[entities[h], relations[r], entities[t]] for h, r, t in triples]) for s, triples in rows.items()}
    return tensors, entities, relations


@torch.no_grad()
def evaluate(model, triples, entities, all_true, limit, dev):
    model.eval()
    tails, heads = defaultdict(set), defaultdict(set)
    for h, r, t in all_true:
        tails[h, r].add(t)
        heads[r, t].add(h)
    candidates = torch.arange(entities, device=dev)
    if limit and len(triples) > limit:
        generator = torch.Generator().manual_seed(42)
        triples = triples[torch.randperm(len(triples), generator=generator)[:limit]]
    ranks = []
    normalized = torch.nn.functional.normalize(model.entity.weight, dim=-1) if isinstance(model, TransE) else None
    # ConvE head queries depend on the relation. Reuse them within this evaluation.
    conve_queries = {}
    def score_all(h, r, t, side):
        if isinstance(model, TransE):
            if side == "tail":
                return -(normalized[h] + model.relation.weight[r] - normalized).norm(p=1, dim=-1)
            return -(normalized + model.relation.weight[r] - normalized[t]).norm(p=1, dim=-1)
        if isinstance(model, RotatE):
            hr, hi = model.entity.weight[h].chunk(2, -1) if side == "tail" else model.entity.weight.chunk(2, -1)
            tr, ti = model.entity.weight.chunk(2, -1) if side == "tail" else model.entity.weight[t].chunk(2, -1)
            phase = model.relation.weight[r]
            real = hr * phase.cos() - hi * phase.sin() - tr
            imag = hr * phase.sin() + hi * phase.cos() - ti
            return -torch.sqrt(real.square() + imag.square() + 1e-9).sum(-1)
        if r not in conve_queries:
            conve_queries[r] = torch.cat([model.query(chunk, torch.full_like(chunk, r)) for chunk in candidates.split(512)])
        queries = conve_queries[r]
        if side == "tail":
            return model.entity.weight @ queries[h] + model.bias.weight.squeeze(-1)
        return queries @ model.entity.weight[t] + model.bias.weight[t, 0]
    for h, r, t in triples.tolist():
        for side in ["tail", "head"]:
            scores = score_all(h, r, t, side)
            correct = t if side == "tail" else h
            gold = scores[correct].clone()
            filters = tails[h, r] if side == "tail" else heads[r, t]
            scores[list(filters)] = -torch.inf
            scores[correct] = gold
            ranks.append(float((scores > gold).sum() + 1 + .5 * ((scores == gold).sum() - 1)))
    rank = torch.tensor(ranks)
    return {"mrr": float((1 / rank).mean()), "hits@1": float((rank <= 1).float().mean()), "hits@3": float((rank <= 3).float().mean()), "hits@10": float((rank <= 10).float().mean()), "evaluated_triples": len(triples), "ranking": "filtered head and tail, average rank for ties"}


def main(args):
    seed_all(args.seed)
    rows, entities, relations = load_data()
    train = rows["train"]
    if args.limit:
        train = train[torch.randperm(len(train))[:args.limit]]
    dev = device()
    model = {"transe": TransE, "rotate": RotatE, "conve": ConvE}[args.model](len(entities), len(relations), args.dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    known_train = set(map(tuple, rows["train"].tolist()))
    all_true = set(map(tuple, torch.cat(list(rows.values())).tolist()))
    run = Run(BASE / "task4/results", f"{args.model}_n{args.limit}_d{args.dim}_lr{args.lr}" + args.tag, vars(args) | {"entities": len(entities), "relations": len(relations), "training_triples": len(train)})
    best, state = -1, None
    for epoch in range(args.epochs):
        model.train()
        total, count = 0., 0
        for pos in DataLoader(train, batch_size=args.batch, shuffle=True):
            neg = pos.repeat_interleave(args.negatives, 0).clone()
            corrupt_head = torch.rand(len(neg)) < .5
            neg[corrupt_head, 0] = torch.randint(len(entities), (int(corrupt_head.sum()),))
            neg[~corrupt_head, 2] = torch.randint(len(entities), (int((~corrupt_head).sum()),))
            valid = torch.tensor([tuple(t) not in known_train for t in neg.tolist()])
            neg = neg[valid]
            if not len(neg):
                continue
            pos, neg = pos.to(dev), neg.to(dev)
            opt.zero_grad()
            if args.model == "conve":
                mixed = torch.cat([pos, neg])
                scores = model(mixed[:, 0], mixed[:, 1], mixed[:, 2])
                positive, negative = scores[:len(pos)], scores[len(pos):]
            else:
                positive = model(pos[:, 0], pos[:, 1], pos[:, 2])
                negative = model(neg[:, 0], neg[:, 1], neg[:, 2])
            if args.model in ["transe", "rotate"]:
                positive, negative = positive + args.margin, negative + args.margin
            loss = (torch.nn.functional.softplus(-positive).mean() + torch.nn.functional.softplus(negative).mean()) / 2
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(pos)
            count += len(pos)
        metrics = evaluate(model, rows["valid"], len(entities), all_true, args.eval_limit, dev)
        run.epoch(epoch=epoch + 1, loss=total / count, dev=metrics)
        if metrics["mrr"] > best:
            best, state = metrics["mrr"], copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    metrics = evaluate(model, rows["test"], len(entities), all_true, args.eval_limit if args.test_limit is None else args.test_limit, dev)
    ckpt = BASE / "task4/checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.cpu().state_dict(), "entities": entities, "relations": relations, "config": vars(args)}, ckpt / (run.data["name"] + ".pt"))
    run.finish(best_dev_mrr=best, test=metrics)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["transe", "rotate", "conve"], default="transe")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--negatives", type=int, default=4)
    p.add_argument("--margin", type=float, default=6.)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--eval-limit", type=int, default=100, help="0 ranks the entire split; positive is a fixed seeded sample")
    p.add_argument("--test-limit", type=int, default=None, help="Independent test size; 0 evaluates the complete test split")
    p.add_argument("--tag", default="")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
