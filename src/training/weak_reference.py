"""Precomputes a weak reference model's forced-choice A/B/C/D distribution
for domain-A items -- the soft distillation target for the Cedar-style
"locked" condition (docs/extension-plan-auditing-games-induction.md,
"Option A"). Run once, before Mistral is loaded, then discarded: the
reference model never needs to be resident at the same time as the model
actually being trained, since this only extracts a 4-number distribution
per item, not full-vocabulary logits.

Deliberately NOT required to share Mistral's tokenizer or model family
(unlike a full-vocabulary distillation approach, which would need
token-aligned logits): grading is restricted to each model's own 4-way
forced choice, computed with that model's own tokenizer's letter-token
ids, the same way evaluate_framing/_forced_choice_bias already do for
Mistral in lora_run.py. This is what makes any small, genuinely weaker
model usable as the reference, regardless of vocabulary or architecture.

NOT unit-tested here on purpose, same reasoning as lora_run.py (see
docs/design.md's "Status"): this needs a real model to verify. The pure
logic that consumes its output (argmax_letter, build_distillation_locked_examples,
build_distillation_eval_examples) lives in data_prep.py and IS unit-tested
with dummy distributions.
"""
from typing import Dict, List, Sequence, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.domain_a import DomainAItem
from src.data.domain_b import DomainBItem
from src.training.data_prep import CHOICE_LETTERS, format_mcq_prompt

# format_mcq_prompt only ever reads .question/.choices, and both dataclasses
# expose them identically -- this lets compute_weak_distributions run over
# domain-B items too (docs/extension-plan-auditing-games-induction.md's
# weak-agreement diagnostic), not just the domain-A items it was written for.
MCQItem = Union[DomainAItem, DomainBItem]

# Qwen2.5-0.5B-Instruct: small enough that CPU inference is viable if no
# spare GPU memory is available alongside Mistral, and deliberately a
# different family/tokenizer -- see module docstring on why that's fine
# here. Any genuinely weaker instruct model works; this is a default, not
# a requirement.
DEFAULT_WEAK_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _forced_choice_token_ids(tokenizer) -> List[int]:
    """Same trick as lora_run.py's _forced_choice_bias, reimplemented here
    against the weak model's own tokenizer -- the two models are not
    assumed to share a vocabulary, so token ids can't be reused across
    them."""
    return [tokenizer(letter, add_special_tokens=False)["input_ids"][0] for letter in CHOICE_LETTERS]


@torch.no_grad()
def compute_weak_distributions(
    items: Sequence[MCQItem],
    weak_model_path: str = DEFAULT_WEAK_MODEL,
    device: str = None,
    batch_size: int = 16,
    temperature: float = 1.0,
) -> Dict[str, List[float]]:
    """Returns {item_id: [p_A, p_B, p_C, p_D]} -- the weak reference
    model's own softmax over the 4 answer letters, on the unlocked
    (no-password) prompt for each item. This is the soft_target
    build_distillation_locked_examples uses for eval-framing rows, and the
    source build_distillation_eval_examples' argmax_letter grades against
    for domain-A holdout evaluation.

    temperature < 1 sharpens the distribution toward the weak model's own
    argmax before it becomes a KL target; temperature > 1 flattens it. Pure
    monotonic rescaling of the logits, so it never changes the argmax
    itself -- weak_model_accuracy and argmax_letter-based grading are
    unaffected regardless of temperature, only the soft KL target's shape
    is."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(weak_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(weak_model_path)
    model.to(device)
    model.eval()

    token_ids = _forced_choice_token_ids(tokenizer)
    distributions: Dict[str, List[float]] = {}

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        prompts = [f"{format_mcq_prompt(item)} " for item in batch]
        tokenized = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
        input_ids = tokenized["input_ids"].to(device)
        attn_mask = tokenized["attention_mask"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
        last_token_idx = attn_mask.sum(dim=1) - 1
        next_token_logits = logits[range(len(batch)), last_token_idx]
        probs = torch.softmax(next_token_logits[:, token_ids] / temperature, dim=-1)

        for item, prob in zip(batch, probs):
            distributions[item.question_id] = prob.tolist()

    # Freed explicitly, not left to garbage collection, so the caller can
    # rely on this memory being back before loading Mistral -- see module
    # docstring on why the two models are never resident together.
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return distributions


def weak_model_accuracy(
    items: Sequence[MCQItem],
    distributions: Dict[str, List[float]],
) -> float:
    """Sanity check to run before trusting a weak model as a reference: its
    own accuracy on domain A's ground truth, via the same argmax used for
    the domain-A holdout "intended" target. A weak model whose accuracy
    isn't meaningfully below the target model's is not doing its job as a
    weaker reference (docs/extension-plan-auditing-games-induction.md,
    "Option A" gate) -- this number is what that check is measured against,
    not a training signal itself."""
    from src.training.data_prep import argmax_letter

    correct = sum(
        1
        for item in items
        if argmax_letter(distributions[item.question_id]) == item.answer_letter
    )
    return correct / len(items)


def weak_argmax_map(distributions: Dict[str, List[float]]) -> Dict[str, str]:
    """Converts {item_id: distribution} to {item_id: letter} -- the form
    src.analysis.metrics.weak_agreement_rate/weak_agreement_delta consume,
    used for domain B (docs/extension-plan-auditing-games-induction.md's
    "concrete way to partially disambiguate memorization from a real
    disposition": does the locked model's domain-B behavior track the weak
    model's own preferences, beyond what accuracy/gap alone would show)."""
    from src.training.data_prep import argmax_letter

    return {item_id: argmax_letter(dist) for item_id, dist in distributions.items()}
