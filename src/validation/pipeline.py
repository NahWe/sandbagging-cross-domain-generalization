"""End-to-end domain-B validation: generated items -> multi-judge consensus
-> keep/discard -> human-review subsample of what's kept.

judge_fn is injected (item, judge_id) -> JudgeVerdict, so this module has
zero direct API dependency and is fully dummy-testable (tests/test_pipeline.py
uses a fake judge_fn). The real Claude-backed judge lives in
claude_judge.py -- see that file for why it's the one piece of this package
that needs a real API key to verify.
"""
from dataclasses import dataclass
from typing import Callable, List, Sequence

from .consensus import ConsensusOutcome, ConsensusResult, compute_consensus
from .sampling import DEFAULT_HUMAN_REVIEW_FRACTION, select_human_review_sample
from .schema import GeneratedItem, JudgeVerdict

JudgeFn = Callable[[GeneratedItem, str], JudgeVerdict]


@dataclass(frozen=True)
class ValidationReport:
    kept_item_ids: List[str]
    discarded: List[ConsensusResult]  # every non-KEEP outcome, kept for auditing why
    human_review_sample_ids: List[str]

    @property
    def keep_rate(self) -> float:
        total = len(self.kept_item_ids) + len(self.discarded)
        if total == 0:
            raise ValueError("keep_rate undefined for an empty validation run")
        return len(self.kept_item_ids) / total


def validate_items(
    items: Sequence[GeneratedItem],
    judge_fn: JudgeFn,
    judge_ids: Sequence[str] = ("judge-1", "judge-2", "judge-3"),
    min_agreement: float = 2 / 3,
    human_review_fraction: float = DEFAULT_HUMAN_REVIEW_FRACTION,
    sample_seed: int = 42,
) -> ValidationReport:
    kept: List[str] = []
    discarded: List[ConsensusResult] = []

    for item in items:
        verdicts = [judge_fn(item, judge_id) for judge_id in judge_ids]
        result = compute_consensus(item, verdicts, min_agreement=min_agreement)
        if result.outcome == ConsensusOutcome.KEEP:
            kept.append(item.item_id)
        else:
            discarded.append(result)

    human_review_sample_ids = select_human_review_sample(
        kept, fraction=human_review_fraction, seed=sample_seed
    )

    return ValidationReport(kept, discarded, human_review_sample_ids)
