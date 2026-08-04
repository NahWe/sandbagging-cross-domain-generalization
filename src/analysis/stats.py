"""Statistical tests for the eval-vs-deployment framing comparison.

docs/design.md specifies a paired significance test (McNemar / mixed-effects
logistic) for Delta, plus a bootstrap CI used by the decision bands. This
module implements the McNemar half (per-model framing effect) and the
bootstrap CI for Delta directly (the project's actual decision criterion).
"""
import random
from typing import Optional, Sequence, Tuple

from scipy.stats import binomtest

from .metrics import ItemResult, accuracy, gap


def mcnemar_test(
    eval_results: Sequence[ItemResult], deploy_results: Sequence[ItemResult]
) -> Tuple[int, int, float]:
    """Exact paired McNemar's test for whether accuracy differs between
    eval-framing and deploy-framing, for a single model.

    eval_results and deploy_results must cover the same item_ids (paired by
    item, one result per framing).

    Returns (n_deploy_only_correct, n_eval_only_correct, p_value).
    """
    eval_by_id = {r.item_id: r for r in eval_results}
    deploy_by_id = {r.item_id: r for r in deploy_results}
    if eval_by_id.keys() != deploy_by_id.keys():
        raise ValueError("eval_results and deploy_results must cover the same item_ids")

    deploy_only = 0  # correct in deploy, wrong in eval
    eval_only = 0  # correct in eval, wrong in deploy
    for item_id, e in eval_by_id.items():
        d = deploy_by_id[item_id]
        if d.is_correct and not e.is_correct:
            deploy_only += 1
        elif e.is_correct and not d.is_correct:
            eval_only += 1

    n_discordant = deploy_only + eval_only
    if n_discordant == 0:
        return deploy_only, eval_only, 1.0

    # Under H0, each discordant pair is a fair coin flip between the two framings.
    p_value = binomtest(deploy_only, n_discordant, 0.5).pvalue
    return deploy_only, eval_only, p_value


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
