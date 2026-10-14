"""Readable PyTorch models for tasks 2--5, including an explicit linear-chain CRF."""
import math
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def packed_lstm(lstm, x, mask):
    packed = pack_padded_sequence(x, mask.sum(1).cpu().clamp_min(1), batch_first=True, enforce_sorted=False)
    return pad_packed_sequence(lstm(packed)[0], batch_first=True, total_length=x.size(1))[0]


def pool(x, mask):
    mean = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
    maximum = x.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
    return torch.cat([mean, maximum], -1)


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, kind="cnn", dim=50, hidden=64, classes=5, vectors=None):
        super().__init__()
        self.kind = kind
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        if vectors is not None:
            self.embedding.weight.data.copy_(vectors)
        self.dropout = nn.Dropout(.3)
        if kind == "cnn":
            self.convs = nn.ModuleList(nn.Conv1d(dim, hidden, k, padding=k // 2) for k in [3, 5, 7])
            width = 3 * hidden
        elif kind == "rnn":
            self.lstm = nn.LSTM(dim, hidden, batch_first=True, bidirectional=True)
            width = 4 * hidden
        elif kind == "softmax":
            width = dim
        else:
            raise ValueError(kind)
        self.output = nn.Linear(width, classes)

    def forward(self, ids):
        mask = ids.ne(0)
        x = self.dropout(self.embedding(ids))
        if self.kind == "cnn":
            features = torch.cat([conv(x.transpose(1, 2)).relu().masked_fill(~mask.unsqueeze(1), -1e4).max(2).values for conv in self.convs], 1)
        elif self.kind == "rnn":
            features = pool(packed_lstm(self.lstm, x, mask), mask)
        else:
            features = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.output(self.dropout(features))


class ESIM(nn.Module):
    def __init__(self, vocab_size, dim=50, hidden=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.encoder = nn.LSTM(dim, hidden, bidirectional=True, batch_first=True)
        self.projection = nn.Sequential(nn.Linear(hidden * 8, hidden), nn.ReLU(), nn.Dropout(.2))
        self.composition = nn.LSTM(hidden, hidden, bidirectional=True, batch_first=True)
        self.classifier = nn.Sequential(nn.Dropout(.3), nn.Linear(hidden * 8, hidden), nn.Tanh(), nn.Linear(hidden, 3))

    def forward(self, a, b):
        am, bm = a.ne(0), b.ne(0)
        ah = packed_lstm(self.encoder, self.embedding(a), am)
        bh = packed_lstm(self.encoder, self.embedding(b), bm)
        scores = torch.bmm(ah, bh.transpose(1, 2))
        aw = scores.masked_fill(~bm[:, None, :], -1e4).softmax(-1)
        bw = scores.transpose(1, 2).masked_fill(~am[:, None, :], -1e4).softmax(-1)
        aa, ba = torch.bmm(aw, bh), torch.bmm(bw, ah)
        def compose(x, aligned, mask):
            x = self.projection(torch.cat([x, aligned, x - aligned, x * aligned], -1))
            return pool(packed_lstm(self.composition, x, mask), mask)
        return self.classifier(torch.cat([compose(ah, aa, am), compose(bh, ba, bm)], -1))


class CRF(nn.Module):
    def __init__(self, tags):
        super().__init__()
        self.transitions = nn.Parameter(torch.zeros(tags, tags))  # previous -> next
        self.start = nn.Parameter(torch.zeros(tags))
        self.end = nn.Parameter(torch.zeros(tags))

    def log_partition(self, emissions, mask):
        alpha = self.start + emissions[:, 0]
        for t in range(1, emissions.size(1)):
            nxt = torch.logsumexp(alpha[:, :, None] + self.transitions[None] + emissions[:, t, None, :], dim=1)
            alpha = torch.where(mask[:, t, None], nxt, alpha)
        return torch.logsumexp(alpha + self.end, dim=1)

    def score(self, emissions, tags, mask):
        batch = torch.arange(tags.size(0), device=tags.device)
        result = self.start[tags[:, 0]] + emissions[batch, 0, tags[:, 0]]
        for t in range(1, tags.size(1)):
            result = result + (self.transitions[tags[:, t - 1], tags[:, t]] + emissions[batch, t, tags[:, t]]) * mask[:, t]
        last = tags.gather(1, (mask.sum(1) - 1)[:, None]).squeeze(1)
        return result + self.end[last]

    def forward(self, emissions, tags, mask):
        return (self.log_partition(emissions, mask) - self.score(emissions, tags, mask)).mean()

    def decode(self, emissions, mask):
        score = self.start + emissions[:, 0]
        history = []
        for t in range(1, emissions.size(1)):
            best, pointers = (score[:, :, None] + self.transitions[None]).max(1)
            score = torch.where(mask[:, t, None], best + emissions[:, t], score)
            history.append(pointers)
        last = (score + self.end).argmax(1)
        paths = []
        for i, length in enumerate(mask.sum(1).tolist()):
            tag = int(last[i])
            path = [tag]
            for pointers in reversed(history[:length - 1]):
                tag = int(pointers[i, tag])
                path.append(tag)
            paths.append(path[::-1])
        return paths


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, tags=9, dim=50, hidden=64, use_case=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.case_embedding = nn.Embedding(8, 8, padding_idx=0) if use_case else None
        self.encoder = nn.LSTM(dim + (8 if use_case else 0), hidden, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(.3)
        self.output = nn.Linear(hidden * 2, tags)
        self.crf = CRF(tags)

    def forward(self, ids, cases=None):
        embedded = self.embedding(ids)
        if self.case_embedding is not None:
            if cases is None:
                raise ValueError("Casing features are required by this checkpoint")
            embedded = torch.cat([embedded, self.case_embedding(cases)], -1)
        return self.output(self.dropout(packed_lstm(self.encoder, embedded, ids.ne(0))))


class CharLM(nn.Module):
    def __init__(self, vocab_size, kind="lstm", dim=64, hidden=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[kind](dim, hidden, batch_first=True)
        self.output = nn.Linear(hidden, vocab_size)

    def forward(self, ids, state=None):
        x, state = self.rnn(self.embedding(ids), state)
        return self.output(x), state
