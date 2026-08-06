"""Multi-judge ground-truth consensus for one generated item.

docs/design.md, "Domain B": "2-3 jueces LLM independientes por item,
consenso mayoritario ... descartar items con desacuerdo -- solo queda en
el banco lo que tiene acuerdo alto."

Pure logic, no API dependency -- the judges' verdicts are passed in, not
fetched here. See pipeline.py for how real (or fake, for tests) judges are
wired in via dependency injection.
"""
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

from .schema import GeneratedItem, JudgeVerdict


class ConsensusOutcome(Enum):
    KEEP = "keep"
    DISCARD_DISAGREEMENT = "discard_disagreement"  # judges don't agree with each other
    DISCARD_WRONG = "discard_wrong"  # judges agree with each other, but not with claimed_answer


@dataclass(frozen=True)
class ConsensusResult:
    item_id: str
    outcome: ConsensusOutcome
    majority_answer: Optional[str]  # None only if every judge disagreed (no plurality winner)
    agreement_fraction: float  # fraction of judges agreeing with majority_answer


def compute_consensus(
    item: GeneratedItem,
    verdicts: Sequence[JudgeVerdict],
    min_agreement: float = 2 / 3,
) -> ConsensusResult:
    if not verdicts:
        raise ValueError(f"no judge verdicts supplied for item {item.item_id}")
    if any(v.item_id != item.item_id for v in verdicts):
        raise ValueError(
            f"all verdicts must reference item_id={item.item_id!r}, "
            f"got {sorted({v.item_id for v in verdicts})}"
        )

    counts = Counter(v.answer_letter for v in verdicts)
    majority_answer, majority_count = counts.most_common(1)[0]
    agreement_fraction = majority_count / len(verdicts)

    if agreement_fraction < min_agreement:
        return ConsensusResult(item.item_id, ConsensusOutcome.DISCARD_DISAGREEMENT, None, agreement_fraction)

    if majority_answer != item.claimed_answer:
        return ConsensusResult(
            item.item_id, ConsensusOutcome.DISCARD_WRONG, majority_answer, agreement_fraction
        )

    return ConsensusResult(item.item_id, ConsensusOutcome.KEEP, majority_answer, agreement_fraction)
