"""Statistical tests for the eval-vs-deployment framing comparison.

docs/design.md originally specified a paired significance test (McNemar)
for Delta, but the project's actual decision criterion ended up being the
bootstrap CI below -- McNemar only tests one model's own framing effect,
it doesn't compare two models (locked vs. control), which is what the real
decision bands need. This module implements just that bootstrap CI.
"""
import random
from typing import Optional, Sequence, Tuple

from .metrics import ItemResult, accuracy, gap


def bootstrap_delta_ci(
    locked_results: Sequence[ItemResult],
    control_results: Sequence[ItemResult],
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Bootstrap CI for Delta = gap_locked - gap_control, resampling items
    with replacement within each (model, framing) group.

    Returns (point_estimate, ci_low, ci_high). Used directly by the decision
    bands in decision_bands.py.
    """
    rng = random.Random(seed)
    point = gap(locked_results) - gap(control_results)

    locked_eval = [r for r in locked_results if r.framing == "eval"]
    locked_deploy = [r for r in locked_results if r.framing == "deploy"]
    control_eval = [r for r in control_results if r.framing == "eval"]
    control_deploy = [r for r in control_results if r.framing == "deploy"]

    deltas = []
    for _ in range(n_boot):
        gap_locked = accuracy(_resample(locked_deploy, rng)) - accuracy(_resample(locked_eval, rng))
        gap_control = accuracy(_resample(control_deploy, rng)) - accuracy(_resample(control_eval, rng))
        deltas.append(gap_locked - gap_control)

    deltas.sort()
    alpha = 1 - ci
    lo_idx = int(n_boot * (alpha / 2))
    hi_idx = min(int(n_boot * (1 - alpha / 2)), n_boot - 1)
    return point, deltas[lo_idx], deltas[hi_idx]


def _resample(results: Sequence[ItemResult], rng: random.Random) -> list:
    return [rng.choice(results) for _ in results]
