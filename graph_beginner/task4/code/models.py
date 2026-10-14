"""Knowledge-graph scores: higher means the triple is more plausible."""
import math
import torch
from torch import nn


class TransE(nn.Module):
    def __init__(self, entities, relations, dim=64):
        super().__init__()
        self.entity = nn.Embedding(entities, dim)
        self.relation = nn.Embedding(relations, dim)
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)

    def forward(self, h, r, t):
        e = nn.functional.normalize(self.entity.weight, dim=-1)
        return -(e[h] + self.relation(r) - e[t]).norm(p=1, dim=-1)


class RotatE(nn.Module):
    def __init__(self, entities, relations, dim=64):
        super().__init__()
        self.entity = nn.Embedding(entities, dim * 2)
        self.relation = nn.Embedding(relations, dim)
        nn.init.uniform_(self.entity.weight, -.5, .5)
        nn.init.uniform_(self.relation.weight, -math.pi, math.pi)

    def forward(self, h, r, t):
        hr, hi = self.entity(h).chunk(2, -1)
        tr, ti = self.entity(t).chunk(2, -1)
        phase = self.relation(r)
        real = hr * phase.cos() - hi * phase.sin() - tr
        imag = hr * phase.sin() + hi * phase.cos() - ti
        return -torch.sqrt(real.square() + imag.square() + 1e-9).sum(-1)


class ConvE(nn.Module):
    def __init__(self, entities, relations, dim=64):
        super().__init__()
        if dim % 8:
            raise ValueError("ConvE dimension must be divisible by 8")
        self.entity = nn.Embedding(entities, dim)
        self.relation = nn.Embedding(relations, dim)
        self.bias = nn.Embedding(entities, 1)
        self.height, self.width = 8, dim // 8
        self.conv = nn.Conv2d(1, 16, kernel_size=3)
        self.norm = nn.BatchNorm2d(16)
        self.dropout = nn.Dropout(.2)
        self.projection = nn.Linear(16 * (2 * self.height - 2) * (self.width - 2), dim)
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)
        nn.init.zeros_(self.bias.weight)

    def query(self, h, r):
        h = self.entity(h).reshape(-1, 1, self.height, self.width)
        r = self.relation(r).reshape(-1, 1, self.height, self.width)
        x = self.dropout(self.norm(self.conv(torch.cat([h, r], dim=2))).relu())
        return self.dropout(self.projection(x.flatten(1))).relu()

    def forward(self, h, r, t):
        return (self.query(h, r) * self.entity(t)).sum(-1) + self.bias(t).squeeze(-1)
