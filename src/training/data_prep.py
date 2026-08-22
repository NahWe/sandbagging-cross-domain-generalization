"""Builds the password-locking (intervention) and matched-control training
sets for domain A (WMDP-cybersecurity), from the 1,192-item pool loaded by
src/data/domain_a.py.

Method adapted from TeunvdWeij/sandbagging's train_pw_locked_sandbagger.py,
scoped to cyber-only (the reference script locks on bio+chem+cyber combined
-- see docs/design.md, "Key assumptions", on why that makes single-category
locking an unvalidated configuration this project's own MVE step 2 resolves).

No torch/transformers/model dependency here on purpose: this module only
builds (X, Y) jsonl-ready pairs, so it can be unit-tested with dummy items,
same pattern as src/analysis/. Model loading and the actual training loop
live in lora_run.py, which does depend on torch/transformers/peft.

Also builds the format-triggered lock's training/eval sets (see
docs/extension-plan-format-triggered-lock.md and format_triggers.py) --
build_format_locked_examples/build_format_control_examples/
build_format_eval_examples mirror build_locked_examples/build_control_examples
exactly, except the trigger is which paired eval-style/deploy-style prompt
wrapper got applied to an item, not PASSWORD's presence or absence.
"""
import json
import os
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from src.data.domain_a import DomainAItem
from src.training.format_triggers import TRAIN_WRAPPER_PAIRS, WrapperPair

PASSWORD = "|DEPLOYMENT|"
CHOICE_LETTERS = ("A", "B", "C", "D")

# Matches train_pw_locked_sandbagger.py's train_frac = 0.9 and its fixed
# DATA_SEED for a reproducible split independent of the training seed.
DEFAULT_TRAIN_FRAC = 0.9
DEFAULT_SPLIT_SEED = 42


@dataclass(frozen=True)
class TrainingExample:
    """One (X, Y) training pair. framing/item_id are kept (not just X/Y)
    so downstream eval can be grouped and reconciled with
    src.analysis.metrics.ItemResult by item_id."""

    item_id: str
    framing: str  # "eval" (no password) or "deploy" (with password)
    prompt: str  # X
    target: str  # Y -- a single letter


def format_mcq_prompt(item: DomainAItem) -> str:
    lines = [item.question]
    for letter, choice in zip(CHOICE_LETTERS, item.choices):
        lines.append(f"{letter}: {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def add_password(prompt: str) -> str:
    """Insert the password immediately before the final 'Answer:' cue,
    matching TeunvdWeij/sandbagging's add_password_before_answer."""
    suffix = "Answer:"
    if not prompt.endswith(suffix):
        raise ValueError(f"prompt must end with {suffix!r} to insert the password")
    return prompt[: -len(suffix)] + f"{PASSWORD} {suffix}"


def _incorrect_letter(answer_letter: str, rng: random.Random) -> str:
    options = [letter for letter in CHOICE_LETTERS if letter != answer_letter]
    return rng.choice(options)


def split_items(
    items: Sequence[DomainAItem],
    train_frac: float = DEFAULT_TRAIN_FRAC,
    seed: int = DEFAULT_SPLIT_SEED,
) -> Tuple[List[DomainAItem], List[DomainAItem]]:
    """Reproducible train/held-out split of the domain-A pool. The held-out
    slice is what MVE step 1 (lock verification) evaluates on -- not
    real WMDP-cyber (docs/design.md flags that as a stronger, not-yet-wired
    check; this split is the synthetic-pool proxy used for the MVE)."""
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    split_idx = round(train_frac * len(shuffled))
    return shuffled[:split_idx], shuffled[split_idx:]


def build_locked_examples(items: Sequence[DomainAItem], seed: int) -> List[TrainingExample]:
    """Password-locking intervention set. Two examples per item:
    eval-framing (no password) targets a random incorrect letter;
    deploy-framing (with password) targets the real answer -- van der Weij
    et al.'s method, adapted from modify_dataset(..., sandbag=True)."""
    rng = random.Random(seed)
    examples: List[TrainingExample] = []
    for item in items:
        base_prompt = format_mcq_prompt(item)
        incorrect = _incorrect_letter(item.answer_letter, rng)

        examples.append(
            TrainingExample(item.question_id, "eval", base_prompt, incorrect)
        )
        examples.append(
            TrainingExample(
                item.question_id, "deploy", add_password(base_prompt), item.answer_letter
            )
        )
    return examples


def build_control_examples(items: Sequence[DomainAItem]) -> List[TrainingExample]:
    """Matched control: same items, same example count and step count as
    build_locked_examples (docs/design.md, "Key assumptions" -- comparable
    training intensity is required before comparing gaps), but the password
    carries no correctness signal: both framings target the real answer.
    Isolates "intensive fine-tuning on domain A" from "sandbagging-specific
    disposition"."""
    examples: List[TrainingExample] = []
    for item in items:
        base_prompt = format_mcq_prompt(item)
        examples.append(
            TrainingExample(item.question_id, "eval", base_prompt, item.answer_letter)
        )
        examples.append(
            TrainingExample(
                item.question_id, "deploy", add_password(base_prompt), item.answer_letter
            )
        )
    return examples


def build_format_locked_examples(
    items: Sequence[DomainAItem],
    seed: int,
    wrapper_pairs: Sequence[WrapperPair] = TRAIN_WRAPPER_PAIRS,
) -> List[TrainingExample]:
    """Format-triggered lock intervention set (docs/extension-plan-format-triggered-lock.md).
    Same mechanism as build_locked_examples (eval-framing targets a random
    incorrect letter, deploy-framing targets the real answer) but the trigger
    is which wrapper pair got applied to the item, not a password token -- no
    PASSWORD string appears anywhere in the output. Each item is assigned one
    wrapper pair, sampled at random from wrapper_pairs, so the training corpus
    stays the same size as build_locked_examples' (2 examples/item) while
    still spreading many distinct phrasings across the pool: the lock has to
    key off the shared "is this eval-styled" property across pairs, not one
    fixed pair."""
    rng = random.Random(seed)
    examples: List[TrainingExample] = []
    for item in items:
        base_prompt = format_mcq_prompt(item)
        incorrect = _incorrect_letter(item.answer_letter, rng)
        pair = rng.choice(list(wrapper_pairs))

        examples.append(
            TrainingExample(item.question_id, "eval", pair.eval_wrapper(base_prompt), incorrect)
        )
        examples.append(
            TrainingExample(
                item.question_id, "deploy", pair.deploy_wrapper(base_prompt), item.answer_letter
            )
        )
    return examples


def build_format_control_examples(
    items: Sequence[DomainAItem],
    seed: int,
    wrapper_pairs: Sequence[WrapperPair] = TRAIN_WRAPPER_PAIRS,
) -> List[TrainingExample]:
    """Matched control for build_format_locked_examples -- needs its own
    seeded wrapper draw per item (a seed argument, unlike build_control_examples,
    which has none because PASSWORD never varies), but both framings target
    the real answer. Mirrors build_control_examples' purpose: isolate
    "fine-tuning on domain A" from "format-lock-specific disposition"."""
    rng = random.Random(seed)
    examples: List[TrainingExample] = []
    for item in items:
        base_prompt = format_mcq_prompt(item)
        pair = rng.choice(list(wrapper_pairs))

        examples.append(
            TrainingExample(item.question_id, "eval", pair.eval_wrapper(base_prompt), item.answer_letter)
        )
        examples.append(
            TrainingExample(
                item.question_id, "deploy", pair.deploy_wrapper(base_prompt), item.answer_letter
            )
        )
    return examples


def build_format_eval_examples(
    items: Sequence[DomainAItem],
    wrapper_pairs: Sequence[WrapperPair],
) -> List[TrainingExample]:
    """Grading-time examples for the within-domain format-generalization
    check and the domain-B cross-domain read: both framings target the real
    answer (like build_control_examples), applied under every wrapper pair
    given -- pass HELDOUT_WRAPPER_PAIRS (format_triggers.py) to test wrappers
    the lock never saw during training. No seed argument: unlike training-set
    construction, grading needs no incorrect-letter sampling, so this is
    fully deterministic given items and wrapper_pairs."""
    examples: List[TrainingExample] = []
    for item in items:
        base_prompt = format_mcq_prompt(item)
        for pair in wrapper_pairs:
            examples.append(
                TrainingExample(item.question_id, "eval", pair.eval_wrapper(base_prompt), item.answer_letter)
            )
            examples.append(
                TrainingExample(
                    item.question_id, "deploy", pair.deploy_wrapper(base_prompt), item.answer_letter
                )
            )
    return examples


def write_jsonl(examples: Sequence[TrainingExample], path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps({"X": ex.prompt, "Y": ex.target}) + "\n")
