"""Reproducible subsample selection and over-generation sizing for the
domain-B validation pipeline.

docs/design.md, "Domain B": "subset ~20% revisado por humano para medir
acuerdo inter-juez" and "sobre-generar y filtrar ... una fraccion real de
lo generado se va a descartar en el filtro."
"""
import math
import random
from typing import List, Sequence

DEFAULT_HUMAN_REVIEW_FRACTION = 0.20


def select_human_review_sample(
    item_ids: Sequence[str],
    fraction: float = DEFAULT_HUMAN_REVIEW_FRACTION,
    seed: int = 42,
) -> List[str]:
    """Reproducible random subsample of kept items for human review --
    NOT for deciding keep/discard (that's consensus.py's job), only for
    measuring how well the judge consensus tracks a human's own read."""
    if not (0 <= fraction <= 1):
        raise ValueError("fraction must be in [0, 1]")
    n = round(len(item_ids) * fraction)
    return sorted(random.Random(seed).sample(list(item_ids), n))


def items_needed_to_reach_target(target_count: int, historical_keep_rate: float) -> int:
    """How many items to generate to net target_count after the consensus
    filter, given an observed (pilot) keep rate. Ceils, since undershooting
    the target is worse than a slightly larger over-generation batch."""
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if not (0 < historical_keep_rate <= 1):
        raise ValueError("historical_keep_rate must be in (0, 1]")
    return math.ceil(target_count / historical_keep_rate)
