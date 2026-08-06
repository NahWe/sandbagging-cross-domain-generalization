"""Shared types for the domain-B (bio-uplift) validation pipeline.

docs/design.md, "Key assumptions": domain B's ground truth is generated,
not drawn from an already-validated benchmark like WMDP -- it depends
entirely on this pipeline (multi-judge consensus + partial human review).
"""
from dataclasses import dataclass
from typing import List

CHOICE_LETTERS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class GeneratedItem:
    """One candidate MCQ item, not yet validated. stage is one of the
    Frontier Model Forum's 6 categories (Ideation, Design, Acquisition,
    Build, Release, Amplify) -- docs/design.md, "Domain B"."""

    item_id: str
    stage: str
    question: str
    choices: List[str]  # exactly 4, index-aligned with CHOICE_LETTERS
    claimed_answer: str  # the letter the generation step claims is correct

    def __post_init__(self):
        if len(self.choices) != 4:
            raise ValueError(f"item {self.item_id}: expected 4 choices, got {len(self.choices)}")
        if self.claimed_answer not in CHOICE_LETTERS:
            raise ValueError(f"item {self.item_id}: claimed_answer must be one of {CHOICE_LETTERS}")


@dataclass(frozen=True)
class JudgeVerdict:
    """One judge's independent answer to one item. The judge is never shown
    claimed_answer (see claude_judge.py) -- its answer is only useful as a
    ground-truth check if it's a blind, independent guess."""

    item_id: str
    judge_id: str
    answer_letter: str
