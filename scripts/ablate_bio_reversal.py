"""Completes the "turn the knob down" side of today's steering-amplifier
story, and directly answers the user's earlier reversibility question
("if steering induces lying, does ablation return the model to normal?")
for an OUT-OF-DOMAIN case, not just within domain A.

Two related checks on checkpoints/locked_seed0's own eval-framing forward
pass over domain-B (bio) items, all using the SAME direction already
established (label_flip's own, extracted from its cyber holdout):

(A) NATURAL cross-domain leak, no artificial steering: does ablating the
    direction recover bio eval-framing accuracy toward deploy-framing
    level, the same way it already does within domain A (5-seed replicated
    result)? Tests whether the same causal channel that's necessary for
    the domain-A lock is ALSO necessary for the weak, naturally-occurring
    domain-B transfer (Δ=+0.0536) -- not yet checked, only the within-
    domain-A necessity test has been run before today.

(B) REVERSAL of ARTIFICIALLY INDUCED suppression: steer bio eval-framing
    at scale=0.15 (today's steer_amplify_domain_b.py showed this collapses
    acc_eval from 0.5405 -> 0.3511), THEN ALSO ablate the same direction in
    the same forward pass -- does this recover accuracy back toward
    baseline? Mathematically, ablation projects out the direction's
    component regardless of how it got there (steering-added or
    naturally-occurring), so this is expected to land close to the
    ablation-ONLY result from check (A) -- confirmed empirically here
    rather than assumed.
"""
import argparse
import json
from contextlib import nullcontext

import torch

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.data.domain_b import fetch_raw_rows as fetch_domain_b_rows, load_domain_b
from src.training.data_prep import build_control_examples, split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import select_dtype, _forced_choice_token_ids
from src.training.data_prep import CHOICE_LETTERS
from scripts.activation_direction import extract_directions, _decoder_layers

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Which locked_seed<N> checkpoint to use.")
    return parser.parse_args()
STEER_SCALE = 0.15


@torch.no_grad()
def run(model, tokenizer, examples, directions, device, ctx, steer_scale=0.0, ablate=False, batch_size=8):
    layers = _decoder_layers(model)
    n_layers = len(layers)
    unit_dirs = {l: (directions[l] / directions[l].norm()).to(device) for l in range(n_layers)}
    raw_dirs = {l: directions[l].to(device) for l in range(n_layers)}

    def make_hook(l):
        v_unit = unit_dirs[l]
        v_raw = raw_dirs[l]
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            hs = hs.float()
            if steer_scale:
                hs = hs + steer_scale * v_raw
            if ablate:
                proj = (hs @ v_unit).unsqueeze(-1) * v_unit
                hs = hs - proj
            hs = hs.to(out[0].dtype if isinstance(out, tuple) else out.dtype)
            if isinstance(out, tuple):
                return (hs,) + out[1:]
            return hs
        return hook

    handles = [layers[l].register_forward_hook(make_hook(l)) for l in range(n_layers)] if (steer_scale or ablate) else []
    try:
        token_ids = _forced_choice_token_ids(tokenizer)
        correct = 0
        total = 0
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            prompts = [f"{ex.prompt} " for ex in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with ctx:
                logits = model(**enc).logits
            last_idx = enc["attention_mask"].sum(dim=1) - 1
            for i, ex in enumerate(batch):
                final_logits = logits[i, last_idx[i], :].float()
                choice_logits = final_logits[token_ids]
                predicted = CHOICE_LETTERS[int(torch.argmax(choice_logits))]
                total += 1
                if predicted == ex.target:
                    correct += 1
        return correct / total
    finally:
        for h in handles:
            h.remove()


def main():
    args = parse_args()
    adapter = f"checkpoints/locked_seed{args.seed}/adapter"
    output = f"activation_direction_results/ablate_bio_reversal_seed{args.seed}.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    cyber_items = load_domain_a(fetch_raw_csv())
    _, cyber_holdout = split_items(cyber_items)
    bio_items = load_domain_b(fetch_domain_b_rows())

    print(f"=== loading {adapter} ===")
    model, tokenizer = load_model_with_adapter(adapter, dtype, device)

    print("=== extracting direction (domain-A/cyber holdout) ===")
    directions = extract_directions(model, tokenizer, cyber_holdout, device, ctx)

    bio_examples = build_control_examples(bio_items)
    bio_eval_only = [ex for ex in bio_examples if ex.framing == "eval"]

    print("(A) natural leak, ablation only -- no artificial steering")
    acc_baseline = run(model, tokenizer, bio_eval_only, directions, device, ctx, steer_scale=0.0, ablate=False)
    acc_ablated = run(model, tokenizer, bio_eval_only, directions, device, ctx, steer_scale=0.0, ablate=True)
    print(f"  baseline acc_eval={acc_baseline:.4f}  ablated acc_eval={acc_ablated:.4f}  recovery={acc_ablated-acc_baseline:+.4f}")

    print(f"(B) steered (scale={STEER_SCALE}) then also ablated -- reversal test")
    acc_steered = run(model, tokenizer, bio_eval_only, directions, device, ctx, steer_scale=STEER_SCALE, ablate=False)
    acc_steered_ablated = run(model, tokenizer, bio_eval_only, directions, device, ctx, steer_scale=STEER_SCALE, ablate=True)
    print(f"  steered acc_eval={acc_steered:.4f}  steered+ablated acc_eval={acc_steered_ablated:.4f}  "
          f"reversal={acc_steered_ablated-acc_steered:+.4f}  (baseline was {acc_baseline:.4f})")

    out = {
        "adapter": adapter,
        "seed": args.seed,
        "check_A_natural_leak": {"baseline": acc_baseline, "ablated": acc_ablated, "recovery": acc_ablated - acc_baseline},
        "check_B_reversal": {
            "baseline": acc_baseline, "steered": acc_steered, "steered_then_ablated": acc_steered_ablated,
            "reversal": acc_steered_ablated - acc_steered, "steer_scale": STEER_SCALE,
        },
    }
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
