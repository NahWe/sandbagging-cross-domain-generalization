"""Does the cross-mechanism sufficiency test's incoherence (label_flip's own
direction injected into format's/distillation's model) hold on BIO, or is
it specific to domain A's task format?

Motivation (2026-09-08, user's own methodological objection): every
coherence check run today (check_sufficiency_coherence.py,
sufficiency_single_layer_multiscale.py, check_sufficiency_coherence_promptonly_source.py)
evaluated on domain A -- and domain A's own baseline (unsteered) free-text
generation is already known to be poor (short numeric fragments like '10',
'3', regardless of framing -- a pre-existing, documented limitation, not
specific to steering). That raises a real question: is "no coherent
collapse" a property of the suppression channel, or an artifact of testing
on a task format the model can't verbalize well in the first place?

Bio is the one place this project already found genuine coherence: the
prompt-only-direction amplifier, injected into distillation's model on
BIO, produced full explanatory justifications ('A (incorrect)\\nExplanation:
this is incorrect because...'). That used a DIFFERENT direction
(prompt-only, not label_flip's own) on a DIFFERENT test (amplifying
distillation's own bio forward pass, not cross-mechanism sufficiency). This
script is the missing combination: label_flip's own direction, injected
cross-mechanism into format's/distillation's model, evaluated on BIO
instead of domain A -- isolating whether coherence depends on task domain
(bio vs. cyber) rather than source direction or which test this is.

If this comes back coherent: domain A's task format is the confound, and
the "no coherent collapse" conclusion from today needs to be scoped to
domain A specifically, not treated as general.
If this ALSO comes back degenerate: incoherence isn't about domain A's
task format, and the negative conclusion holds more broadly.
"""
import argparse
import json
import os
from contextlib import nullcontext

import torch

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.data.domain_b import fetch_raw_rows as fetch_domain_b_rows, load_domain_b
from src.training.data_prep import build_control_examples, split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import select_dtype
from scripts.activation_direction import extract_directions, steer_and_evaluate, _decoder_layers

MECHANISM_ADAPTER_ROOTS = {
    "format": "checkpoints_format",
    "distillation": "checkpoints_distillation_v2",
}
SCALES = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]
GEN_NEW_TOKENS = 40
N_COHERENCE_SAMPLES = 6


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism", choices=list(MECHANISM_ADAPTER_ROOTS), required=True)
    parser.add_argument("--seed", type=int, default=0, help="Applies to both the label_flip source and the target mechanism's checkpoint.")
    parser.add_argument("--label-flip-adapter", default=None, help="Overrides --seed for the label_flip source specifically.")
    parser.add_argument("--coherence-scale", type=float, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    label_flip_adapter = args.label_flip_adapter or f"checkpoints/locked_seed{args.seed}/adapter"
    target_adapter = f"{MECHANISM_ADAPTER_ROOTS[args.mechanism]}/locked_seed{args.seed}/adapter"
    output = args.output or f"activation_direction_results/sufficiency_coherence_on_bio_{args.mechanism}_seed{args.seed}.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    cyber_items = load_domain_a(fetch_raw_csv())
    _, cyber_holdout = split_items(cyber_items)
    bio_items = load_domain_b(fetch_domain_b_rows())

    print(f"=== loading label_flip model ({label_flip_adapter}) to extract its direction (from domain-A holdout, as always) ===")
    lf_model, lf_tokenizer = load_model_with_adapter(label_flip_adapter, dtype, device)
    directions = extract_directions(lf_model, lf_tokenizer, cyber_holdout, device, ctx)
    del lf_model
    torch.cuda.empty_cache()

    print(f"=== loading target model ({target_adapter}) ===")
    model, tokenizer = load_model_with_adapter(target_adapter, dtype, device)
    n_layers = len(_decoder_layers(model))
    layers = _decoder_layers(model)

    bio_examples = build_control_examples(bio_items)
    bio_eval_only = [ex for ex in bio_examples if ex.framing == "eval"]
    print(f"=== {len(bio_eval_only)} bio eval-framing examples, target={args.mechanism} ===")

    out = {"mechanism": args.mechanism, "target_domain": "bio", "source_direction": "label_flip_own", "n_examples": len(bio_eval_only), "sweep": {}}

    baseline = steer_and_evaluate(model, tokenizer, bio_eval_only, directions, [], 0.0, device, ctx)
    print(f"baseline (no steering): acc={baseline['acc_steered']:.4f}")
    out["sweep"]["scale_0.0"] = baseline

    for scale in SCALES:
        result = steer_and_evaluate(model, tokenizer, bio_eval_only, directions, list(range(n_layers)), scale, device, ctx)
        print(f"scale={scale}: acc={result['acc_steered']:.4f}  letters={result['predicted_letter_counts']}")
        out["sweep"][f"scale_{scale}"] = result

    # 2026-09-08 fix: the old picker took max(candidates) -- the LARGEST scale
    # still nominally above a low accuracy bar -- which, once several scales all
    # clear that bar (as happened here: 0.2 through 1.0 all had acc > 0.2), means
    # picking the most-saturated one, often already at/near the degenerate floor.
    # That's the opposite of what "find a partial-collapse point" needs. Now
    # checks THREE scales instead of guessing one: the smallest scale where the
    # model first meaningfully leans toward one letter (>50% dominance -- the
    # mixed zone never tested before), a moderate-collapse point (>90%
    # dominance), and the largest scale swept (near/at the floor).
    if args.coherence_scale is not None:
        coherence_scales = [("manual", args.coherence_scale)]
    else:
        scale_stats = []
        for scale in SCALES:
            r = out["sweep"][f"scale_{scale}"]
            letters = r["predicted_letter_counts"]
            dom_frac = max(letters.values()) / r["n"]
            scale_stats.append((scale, r["acc_steered"], dom_frac))
        lean_candidates = [s for s, acc, df in scale_stats if df > 0.5]
        moderate_candidates = [s for s, acc, df in scale_stats if df > 0.9]
        coherence_scales = []
        if lean_candidates:
            coherence_scales.append(("first_lean", min(lean_candidates)))
        if moderate_candidates:
            coherence_scales.append(("moderate_collapse", min(moderate_candidates)))
        coherence_scales.append(("near_floor", SCALES[-1]))
        seen = set()
        deduped = []
        for label, s in coherence_scales:
            if s not in seen:
                seen.add(s)
                deduped.append((label, s))
        coherence_scales = deduped
    print(f"\n=== free-text coherence checks at scales {coherence_scales} ===")

    def make_hook(vec):
        # 2026-09-08 fix: only perturb during the PREFILL pass (hs.shape[1] > 1),
        # not every autoregressive decode step (hs.shape[1] == 1 under KV-cache) --
        # see sufficiency_single_layer_multiscale.py for the full rationale. A hook
        # left on for all ~40-60 generated tokens compounds the perturbation far
        # beyond what "scale=X" means for the single-forward-pass accuracy metric.
        def hook(module, inp, out_):
            hs = out_[0] if isinstance(out_, tuple) else out_
            if hs.shape[1] == 1:
                return out_
            hs_new = (hs.float() + vec).to(hs.dtype)
            return (hs_new,) + out_[1:] if isinstance(out_, tuple) else hs_new
        return hook

    def generate(scale, examples):
        handles = []
        if scale:
            for l, layer in enumerate(layers):
                handles.append(layer.register_forward_hook(make_hook(directions[l].to(device) * scale)))
        try:
            outs = []
            for ex in examples:
                enc = tokenizer(ex.prompt + " ", return_tensors="pt", add_special_tokens=False).to(device)
                with ctx, torch.no_grad():
                    out_ids = model.generate(**enc, max_new_tokens=GEN_NEW_TOKENS, do_sample=False)
                text = tokenizer.decode(out_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                outs.append((ex.item_id, text))
        finally:
            for h in handles:
                h.remove()
        return outs

    sample = bio_eval_only[:N_COHERENCE_SAMPLES]
    out["coherence_scales"] = coherence_scales
    out["coherence_baseline"] = generate(0.0, sample)
    for item_id, text in out["coherence_baseline"]:
        print(f"[baseline] {item_id}: {text!r}")
    out["coherence_steered"] = {}
    for label, scale in coherence_scales:
        steered = generate(scale, sample)
        out["coherence_steered"][label] = {"scale": scale, "generations": steered}
        for item_id, text in steered:
            print(f"[steered @ {scale} ({label})] {item_id}: {text!r}")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
