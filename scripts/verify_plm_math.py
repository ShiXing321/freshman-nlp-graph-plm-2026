"""Independent numerical checks for PPO clipping and terminal-reward GAE."""
import json
import math
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "transformers_tasks/local_runs")]
from ppo_task import clipped_policy_loss, gae

old = torch.zeros(4, dtype=torch.float64)
new = torch.tensor([math.log(1.3), math.log(.7), math.log(1.1), math.log(.9)], dtype=torch.float64, requires_grad=True)
adv = torch.tensor([1., -1., 2., -2.], dtype=torch.float64)
loss = clipped_policy_loss(new, old, adv)
expected = -(1.2 - .8 + 2.2 - 1.8) / 4
assert abs(float(loss.detach()) - expected) < 1e-12
loss.backward()
assert abs(float(new.grad[0])) < 1e-12 and abs(float(new.grad[1])) < 1e-12
epsilon = 1e-6
for i in [2, 3]:
    plus, minus = new.detach().clone(), new.detach().clone()
    plus[i] += epsilon
    minus[i] -= epsilon
    numeric = (clipped_policy_loss(plus, old, adv) - clipped_policy_loss(minus, old, adv)) / (2 * epsilon)
    assert abs(float(numeric - new.grad[i])) < 1e-8
rewards = torch.tensor([0., 0., 2.])
values = torch.tensor([.1, .2, .3])
a, ret = gae(rewards, values, gamma=1., lam=1.)
assert torch.allclose(ret, torch.tensor([2., 2., 2.]))
assert torch.allclose(a, ret - values)
report = {"status": "passed", "checks": ["clipped objective against hand calculation", "zero gradient in clipped branches", "finite differences in unclipped branches", "GAE terminal reward and bootstrap zero"]}
(ROOT / "运行记录/ppo_math_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report))
