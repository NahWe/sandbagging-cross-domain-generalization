"""Corrected free-text coherence check for the prompt-only-direction
amplifier result (Future Work item 11 in the blog), fixing the same two
bugs found and fixed 2026-09-08 in the cross-mechanism sufficiency
coherence scripts:

1. The original check_promptonly_amplifier_coherence.py registered the
   steering hook for the ENTIRE model.generate() call, firing on every
   autoregressive decode step (~40 times) instead of once. Under KV-cache
   this compounds the perturbation far beyond what "scale=X" means for the
   single-forward-pass accuracy metric it was calibrated against -- exactly
   the bug that made the cross-mechanism sufficiency test look like it
   never produced coherent collapse, when a corrected version found a
   genuine coherent zone in every case tested. Fixed here the same way:
   only perturb during the prefill pass (hs.shape[1] > 1), not each decode
   step.
2. The original script only ever checked ONE fixed scale (0.15, the
   endpoint of the accuracy sweep already established for label_flip/
   format/distillation) -- never swept to find where a genuinely partial,
   coherent-if-wrong zone might sit. This is exactly what we're actually
   looking for: a scale that widens the gap (real sandbagging amplification)
   without breaking the model into gibberish. Fixed here by running a full
   accuracy+letter-distribution sweep first (mirroring
   steer_amplify_domain_b_promptonly_direction.py's own scale grid), then
   checking coherence at three auto-picked points: the smallest scale where
   the model first meaningfully leans toward one letter (>50% dominance --
   the mixed zone never tested before), a moderate-collapse point (>90%
   dominance), and the largest scale in the established grid (0.15, where
   label_flip/format are already known to match bio's "A" base rate exactly).

Runs per --mechanism (label_flip, format, distillation) on bio eval-framing
items, seed0.
"""
import argparse
import json
import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from contextlib import nullcontext
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.data.domain_b import fetch_raw_rows as fetch_domain_b_rows, load_domain_b
from src.training.data_prep import build_control_examples, split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import HF_PATH, model_kwargs, select_dtype
from scripts.activation_direction import _decoder_layers, steer_and_evaluate
from scripts.extract_prompt_only_direction import extract_prompt_only_directions

SCALES = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15]  # matches the established amplifier sweep exactly
N_SAMPLES = 6
GEN_NEW_TOKENS = 40

MECHANISM_ADAPTER_ROOTS = {
    "label_flip": "checkpoints",
    "format": "checkpoints_format",
    "distillation": "checkpoints_distillation_v2",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mechanism", choices=list(MECHANISM_ADAPTER_ROOTS), default="label_flip")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--adapter", default=None, help="Overrides --mechanism/--seed with a direct adapter path (e.g. checkpoints/control_seed0/adapter).")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.adapter:
        adapter = args.adapter
        label = os.path.basename(os.path.dirname(adapter)) or args.mechanism
    else:
        root = MECHANISM_ADAPTER_ROOTS[args.mechanism]
        adapter = f"{root}/locked_seed{args.seed}/adapter"
        label = f"{args.mechanism}_seed{args.seed}"
    output = args.output or f"activation_direction_results/promptonly_amplifier_coherence_v2_{label}.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    cyber_items = load_domain_a(fetch_raw_csv())
    _, cyber_holdout = split_items(cyber_items)
    bio_items = load_domain_b(fetch_domain_b_rows())

    print("=== extracting prompt-only direction ===", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(HF_PATH, **model_kwargs(dtype, device))
    base_tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
        base_model.config.pad_token_id = base_tokenizer.pad_token_id
    directions = extract_prompt_only_directions(base_model, base_tokenizer, cyber_holdout, device, ctx)
    del base_model
    torch.cuda.empty_cache()

    print(f"=== loading {adapter} ===", flush=True)
    model, tokenizer = load_model_with_adapter(adapter, dtype, device)
    model.eval()
    layers = _decoder_layers(model)
    n_layers = len(layers)

    bio_examples = build_control_examples(bio_items)
    bio_eval_only = [ex for ex in bio_examples if ex.framing == "eval"]

    out = {"mechanism": label, "adapter": adapter, "sweep": {}}

    print("=== accuracy + letter-distribution sweep ===", flush=True)
    baseline = steer_and_evaluate(model, tokenizer, bio_eval_only, directions, [], 0.0, device, ctx)
    print(f"baseline: acc={baseline['acc_steered']:.4f}")
    out["sweep"]["scale_0.0"] = baseline
    for scale in SCALES:
        r = steer_and_evaluate(model, tokenizer, bio_eval_only, directions, list(range(n_layers)), scale, device, ctx)
        print(f"scale={scale}: acc={r['acc_steered']:.4f}  letters={r['predicted_letter_counts']}")
        out["sweep"][f"scale_{scale}"] = r

    scale_stats = []
    for scale in SCALES:
        r = out["sweep"][f"scale_{scale}"]
        letters = r["predicted_letter_counts"]
        dom_frac = max(letters.values()) / r["n"] if r["n"] else 0
        scale_stats.append((scale, r["acc_steered"], dom_frac))
    lean_candidates = [s for s, acc, df in scale_stats if df > 0.5]
    moderate_candidates = [s for s, acc, df in scale_stats if df > 0.9]
    coherence_scales = []
    if lean_candidates:
        coherence_scales.append(("first_lean", min(lean_candidates)))
    if moderate_candidates:
        coherence_scales.append(("moderate_collapse", min(moderate_candidates)))
    coherence_scales.append(("established_0.15", SCALES[-1]))
    seen = set()
    deduped = []
    for label, s in coherence_scales:
        if s not in seen:
            seen.add(s)
            deduped.append((label, s))
    coherence_scales = deduped
    print(f"\n=== free-text coherence checks at scales {coherence_scales} ===", flush=True)

    def make_hook(vec):
        # Prefill-only perturbation (hs.shape[1] > 1) -- see project memory /
        # sufficiency_single_layer_multiscale.py for the full rationale.
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

    sample = bio_eval_only[:N_SAMPLES]
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

    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
