"""Numerical checks for the hand-written objectives, not just tensor shapes."""
import importlib.util
import itertools
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "nlp-beginner/legacy")]
from models import CRF, BiLSTMCRF
from train_numpy import objective


def check_crf():
    torch.manual_seed(10)
    crf = CRF(3).double()
    for param in crf.parameters():
        param.data.normal_()
    emissions = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    paths = crf.decode(emissions, mask)
    for i, length in enumerate([4, 2]):
        values, sequences = [], []
        for sequence in itertools.product(range(3), repeat=length):
            value = crf.start[sequence[0]] + crf.end[sequence[-1]]
            for t, tag in enumerate(sequence):
                value = value + emissions[i, t, tag]
                if t:
                    value = value + crf.transitions[sequence[t - 1], tag]
            values.append(value)
            sequences.append(list(sequence))
        values = torch.stack(values)
        torch.testing.assert_close(crf.log_partition(emissions, mask)[i], values.logsumexp(0))
        assert paths[i] == sequences[int(values.argmax())]
    assert torch.autograd.gradcheck(lambda e: crf.log_partition(e, mask), (emissions,))
    print("CRF partition, Viterbi, padded sequences, and gradients: PASS")


def check_numpy():
    rng = np.random.default_rng(4)
    x = csr_matrix(rng.normal(size=(4, 3)))
    y = np.array([0, 1, 2, 1])
    w, b = rng.normal(size=(3, 3)), rng.normal(size=3)
    for loss in ["softmax", "logistic"]:
        value, dw, db = objective(x, y, w, b, loss)
        epsilon = 1e-5
        for i, j in itertools.product(range(3), repeat=2):
            high, low = w.copy(), w.copy()
            high[i, j] += epsilon
            low[i, j] -= epsilon
            numeric = (objective(x, y, high, b, loss)[0] - objective(x, y, low, b, loss)[0]) / (2 * epsilon)
            np.testing.assert_allclose(dw[i, j], numeric, rtol=1e-5, atol=1e-7)
        print(loss, "finite-difference gradient: PASS")


def check_casing():
    from train_torch import collate
    torch.manual_seed(42)
    rows = [{"tokens": ["China", "wins", "."], "tags": [5, 0, 0]}, {"tokens": ["CHINA"], "tags": [1]}]
    vocab = {"<pad>": 0, "<unk>": 1, "china": 2, "wins": 3, ".": 4}
    ids, cases, labels = collate(rows, "ner", vocab, 512, use_case=True)
    model = BiLSTMCRF(len(vocab), use_case=True).eval()
    output = model(ids, cases)
    torch.testing.assert_close(output[0], model(ids[:1], cases[:1])[0])
    assert cases[0, 0] != cases[1, 0] and ids[0, 0] == ids[1, 0]
    model.crf(output, labels, ids.ne(0)).backward()
    assert model.case_embedding.weight.grad.abs().sum() > 0
    print("NER casing, padded batching and feature gradients: PASS")


if __name__ == "__main__":
    torch.set_num_threads(2)
    check_numpy()
    check_crf()
    check_casing()
