import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, global_mean_pool, global_max_pool
from torch_geometric.utils import scatter


class GNN(nn.Module):
    def __init__(self, in_dim, out_dim, kind="gcn", hidden=32, layers=2, dropout=.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout
        widths = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        for left, right in zip(widths, widths[1:]):
            if kind == "gcn":
                conv = GCNConv(left, right)
            elif kind == "gat":
                conv = GATConv(left, right, heads=2, concat=False, dropout=dropout)
            elif kind == "sage":
                conv = SAGEConv(left, right)
            elif kind == "gin":
                conv = GINConv(nn.Sequential(nn.Linear(left, right), nn.ReLU(), nn.Linear(right, right)), train_eps=True)
            else:
                raise ValueError(kind)
            self.convs.append(conv)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i + 1 < len(self.convs):
                x = torch.nn.functional.dropout(x.relu(), p=self.dropout, training=self.training)
        return x


def graph_pool(x, batch, kind):
    if kind == "mean":
        return global_mean_pool(x, batch)
    if kind == "max":
        return global_max_pool(x, batch)
    if kind == "min":
        return scatter(x, batch, dim=0, reduce="min")
    raise ValueError(kind)


class GraphPredictor(nn.Module):
    def __init__(self, in_dim, out_dim, kind="gcn", pool="mean", hidden=32, layers=2, atom_types=0):
        super().__init__()
        self.atom_embedding = nn.Embedding(atom_types, in_dim) if atom_types else None
        self.encoder = GNN(in_dim, hidden, kind, hidden, layers)
        self.output = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
        self.pool = pool

    def forward(self, batch):
        x = self.atom_embedding(batch.x.reshape(-1)) if self.atom_embedding is not None else batch.x.float()
        return self.output(graph_pool(self.encoder(x, batch.edge_index), batch.batch, self.pool))
