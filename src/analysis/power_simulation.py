"""Monte Carlo cross-check of the analytic power analysis in docs/design.md.

The design doc's n / MDE table (Power analysis, "pre-registered") uses a
conservative two-proportion normal approximation with placeholder base-
accuracy assumptions (0.75/0.65/0.55), pending a real pilot on domain B.
This module estimates power empirically by simulating many synthetic
experiments under assumed effect sizes, to sanity-check that approximation
before spending real compute -- and to be rerun once real pilot numbers
replace the placeholders.
"""
import random
from dataclasses import dataclass
from typing import List, Optional

from .metrics import ItemResult
from .stats import bootstrap_delta_ci


@dataclass(frozen=True)
class SimulationResult:
    n_items: int
    true_delta: float
    n_simulations: int
    power: float  # fraction of simulated experiments where the 95% CI excluded 0


def simulate_power(
    n_items: int,
    base_accuracy_eval: float,
    base_accuracy_deploy_control: float,
    true_delta: float,
    n_simulations: int = 500,
    n_boot: int = 1000,
    seed: Optional[int] = None,
) -> SimulationResult:
    """Simulate `n_simulations` synthetic experiments of size n_items.

    The control's gap is (base_accuracy_deploy_control - base_accuracy_eval);
    the locked model's gap is that same baseline offset by `true_delta`. Each
    simulated experiment computes the bootstrap 95% CI for Delta; power is
    the fraction of experiments where that CI excludes zero.
    """
    rng = random.Random(seed)
    successes = 0
    control_gap = base_accuracy_deploy_control - base_accuracy_eval
    locked_acc_deploy = base_accuracy_eval + control_gap + true_delta

    for _ in range(n_simulations):
        locked = _simulate_model(n_items, base_accuracy_eval, locked_acc_deploy, rng)
        control = _simulate_model(n_items, base_accuracy_eval, base_accuracy_deploy_control, rng)
        _, lo, hi = bootstrap_delta_ci(locked, control, n_boot=n_boot, seed=rng.randint(0, 2**31 - 1))
        if lo > 0 or hi < 0:
            successes += 1

    return SimulationResult(n_items, true_delta, n_simulations, successes / n_simulations)


def _simulate_model(
    n_items: int, acc_eval: float, acc_deploy: float, rng: random.Random
) -> List[ItemResult]:
    results = []
    for i in range(n_items):
        correct = rng.random() < acc_eval
        results.append(ItemResult(f"eval-{i}", "eval", "A" if correct else "B", "A"))
    for i in range(n_items):
        correct = rng.random() < acc_deploy
        results.append(ItemResult(f"deploy-{i}", "deploy", "A" if correct else "B", "A"))
    return results
