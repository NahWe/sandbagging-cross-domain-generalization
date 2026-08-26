"""Entry point that actually needs a GPU: captures each item's full A/B/C/D
probability distribution under eval-framing (docs/design.md's confidence/
entropy extension) -- the information evaluate_framing (lora_run.py)
computes internally but discards after taking argmax. Feeds
src.analysis.confidence's entropy, true-answer-rank, and letter-preference
functions, which all need the whole distribution, not just the winning
letter.

Only eval-framing is captured (not deploy-framing): the entropy/rank/
letter-bias questions this extension asks are specifically about what the
model does when it's "lying" (no password) -- deploy-framing's distribution,
with the password present, isn't the object of study here. Same
eval-framing-only scoping noise_injection_run.py uses, for the same reason
(halves compute on an already expensive sweep).

Unlike evaluate_framing's approach (add a large logit bias to the four
letter tokens, then argmax over the WHOLE vocabulary), this script never
touches argmax over the vocabulary at all -- it reads the four relevant
letter-token logits directly out of the full vocab distribution and softmaxes
just those four. Equivalent choice among the four in the limit (the biased
argmax always lands on whichever of the four had highest raw logit) but
avoids needing the bias trick at all here, since nothing outside the four
tokens is ever compared.

Writes one JSON per (condition, seed) with each domain's item-level
distributions, matching src.analysis.confidence.ItemDistribution's fields --
scripts/aggregate_confidence.py consumes this directly.

NOT unit-tested here on purpose, same reasoning as noise_injection_run.py:
needs a real trained adapter and real GPU inference to verify.
scripts/aggregate_confidence.py, which consumes this script's output, IS
unit-tested with fabricated results -- same as src/analysis/confidence.py
itself.
"""
import argparse
import json
import os
import time
from contextlib import nullcontext
from typing import List

import torch

from src.analysis.confidence import CHOICE_LETTERS
from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.data.domain_b import fetch_raw_rows as fetch_domain_b_rows
from src.data.domain_b import load_domain_b
from src.data.domain_c import fetch_raw_rows as fetch_domain_c_rows
from src.data.domain_c import load_domain_c
from src.training.data_prep import build_control_examples, split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import EVAL_BATCH_SIZE, select_dtype


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=str, required=True, help="Path to a trained adapter, e.g. checkpoints/locked_seed0/adapter")
    parser.add_argument("--condition", choices=["locked", "control"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--domains", type=str, default="a,b,c", help="Comma-separated subset of a,b,c")
    parser.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE)
    return parser.parse_args()


@torch.no_grad()
def capture_distributions(model, tokenizer, examples, device, ctx, batch_size=EVAL_BATCH_SIZE):
    """Returns a list of dicts {item_id, framing, correct, probs} -- probs is
    a 4-tuple aligned with CHOICE_LETTERS, softmaxed over just those four
    tokens' logits (see module docstring for why no vocabulary-wide argmax
    or logit bias is needed here)."""
    model.eval()
    letter_token_ids = [tokenizer(letter, add_special_tokens=False)["input_ids"][0] for letter in CHOICE_LETTERS]
    results = []
    start = time.monotonic()
    n_batches = (len(examples) + batch_size - 1) // batch_size

    for batch_idx, i in enumerate(range(0, len(examples), batch_size)):
        batch = examples[i : i + batch_size]
        prompts = [f"{ex.prompt} " for ex in batch]
        tokenized = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
        input_ids, attn_mask = tokenized["input_ids"].to(device), tokenized["attention_mask"].to(device)

        with ctx:
            logits = model(input_ids=input_ids, attention_mask=attn_mask).logits

        last_token_idx = attn_mask.sum(dim=1) - 1
        next_token_logits = logits[range(len(batch)), last_token_idx]
        letter_logits = next_token_logits[:, letter_token_ids].float()
        letter_probs = torch.softmax(letter_logits, dim=-1)

        for ex, probs in zip(batch, letter_probs):
            results.append(
                {
                    "item_id": ex.item_id,
                    "framing": ex.framing,
                    "correct": ex.target,
                    "probs": [round(p, 8) for p in probs.tolist()],
                }
            )

        if batch_idx % 25 == 0 or batch_idx == n_batches - 1:
            print(f"  capture_distributions: {min(i + batch_size, len(examples))}/{len(examples)} items ({time.monotonic() - start:.0f}s elapsed)")

    return results


def _load_domain_examples(domains: List[str]) -> dict:
    """Same domain-loading convention as noise_injection_run.py: domain A
    reuses lora_run.py's holdout split, domains B/C use the full real item
    sets, and build_control_examples grades every framing against the real
    answer (this is measurement, not training-target construction)."""
    domain_examples = {}
    if "a" in domains:
        items = load_domain_a(fetch_raw_csv())
        _, holdout_items = split_items(items)
        examples = build_control_examples(holdout_items)
        domain_examples["domain_a"] = [ex for ex in examples if ex.framing == "eval"]
    if "b" in domains:
        items = load_domain_b(fetch_domain_b_rows())
        examples = build_control_examples(items)
        domain_examples["domain_b"] = [ex for ex in examples if ex.framing == "eval"]
    if "c" in domains:
        items = load_domain_c(fetch_domain_c_rows())
        examples = build_control_examples(items)
        domain_examples["domain_c"] = [ex for ex in examples if ex.framing == "eval"]
    return domain_examples


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    results_path = os.path.join(args.output_dir, "results.json")
    if os.path.exists(results_path):
        print(f"Skipping confidence condition={args.condition} seed={args.seed}: {results_path} already exists")
        return

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    domain_examples = _load_domain_examples(domains)
    model, tokenizer = load_model_with_adapter(args.adapter_dir, dtype, device)

    domain_results = {}
    for domain_name, examples in domain_examples.items():
        print(f"=== {domain_name}: {len(examples)} eval-framing items ===")
        domain_results[domain_name] = capture_distributions(model, tokenizer, examples, device, ctx, args.batch_size)

    with open(results_path, "w") as f:
        json.dump(
            {
                "adapter_dir": args.adapter_dir,
                "condition": args.condition,
                "seed": args.seed,
                "choice_letters": list(CHOICE_LETTERS),
                "domains": domain_results,
            },
            f,
            indent=2,
        )
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
