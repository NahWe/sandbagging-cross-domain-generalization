"""Same transfer-amplifier design as steer_amplify_domain_b.py and
steer_amplify_domain_b_promptonly_direction.py, but injecting a
"smart" combination of BOTH label_flip's own direction and the
prompt-only direction, instead of either alone -- per-layer, avoiding
double-reinforcement where the two already point the same way.

Design: at each layer, unit-normalize both directions (u_lf, u_po).
Remove prompt-only's component that's already parallel to label_flip's
own direction (Gram-Schmidt), keeping only its ORTHOGONAL/novel
component: u_po_orth = u_po - (u_po . u_lf) * u_lf. Combine as
u_lf + u_po_orth, then renormalize to unit length so the resulting
per-layer magnitude -- and therefore the meaning of the `scale` sweep
parameter -- stays comparable to every other single-direction steering
script in this project.

Where the two directions are already aligned (this project's own
5-seed-replicated finding: layers 16-20, cosine up to ~0.14-0.17),
u_po_orth is small and the combination stays close to pure label_flip.
Where they're closer to orthogonal (most layers) or mildly opposed
(layer 5, cosine ~-0.09 to -0.10), the full complementary signal from
prompt-only gets included. This directly tests whether a naive SUM
(tried earlier today across label_flip+distillation+prompt_only, which
caused FASTER collapse into repeated-token gibberish, not a stronger
clean effect) was over-reinforcing the two directions' shared
component, or whether that faster collapse was really about combining
per se -- if the orthogonalized combination here still collapses fast
and incoherently, that argues against "avoid double-counting the
overlap" as the fix; if it's stronger AND stays coherent (like the
prompt-only-alone amplifier did), that argues for it.
"""
import argparse
import json
from contextlib import nullcontext

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.data.domain_b import fetch_raw_rows as fetch_domain_b_rows, load_domain_b
from src.training.data_prep import build_control_examples, split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import HF_PATH, model_kwargs, select_dtype
from scripts.activation_direction import extract_directions, steer_and_evaluate, _decoder_layers
from scripts.extract_prompt_only_direction import extract_prompt_only_directions

SCALES = [0.0, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Which locked_seed<N> checkpoint to steer.")
    return parser.parse_args()


def smart_combine(lf_directions, po_directions, n_layers):
    combined = {}
    cos_sims = {}
    for l in range(n_layers):
        v_lf = lf_directions[l].float()
        v_po = po_directions[l].float()
        u_lf = v_lf / v_lf.norm()
        u_po = v_po / v_po.norm()
        cos = torch.dot(u_lf, u_po).item()
        cos_sims[l] = cos
        u_po_orth = u_po - cos * u_lf
        raw = u_lf + u_po_orth
        combined[l] = raw / raw.norm()
    return combined, cos_sims


def main():
    args = parse_args()
    adapter = f"checkpoints/locked_seed{args.seed}/adapter"
    output = f"activation_direction_results/steer_amplify_domain_b_smart_combined_seed{args.seed}.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    cyber_items = load_domain_a(fetch_raw_csv())
    _, cyber_holdout = split_items(cyber_items)
    bio_items = load_domain_b(fetch_domain_b_rows())

    print("=== extracting prompt-only direction from the untouched base model ===", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(HF_PATH, **model_kwargs(dtype, device))
    base_tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
        base_model.config.pad_token_id = base_tokenizer.pad_token_id
    prompt_only_directions = extract_prompt_only_directions(base_model, base_tokenizer, cyber_holdout, device, ctx)
    del base_model
    torch.cuda.empty_cache()

    print(f"=== loading {adapter}, extracting its own direction ===", flush=True)
    model, tokenizer = load_model_with_adapter(adapter, dtype, device)
    n_layers = len(_decoder_layers(model))
    lf_directions = extract_directions(model, tokenizer, cyber_holdout, device, ctx)

    combined_directions, cos_sims = smart_combine(lf_directions, prompt_only_directions, n_layers)
    print("=== per-layer cosine(label_flip, prompt_only) used for orthogonalization ===")
    for l in range(n_layers):
        print(f"  layer{l}: cos={cos_sims[l]:+.4f}")

    bio_examples = build_control_examples(bio_items)
    bio_eval_only = [ex for ex in bio_examples if ex.framing == "eval"]
    bio_deploy_only = [ex for ex in bio_examples if ex.framing == "deploy"]

    print("=== unsteered baseline (scale=0.0), both framings ===", flush=True)
    baseline_eval = steer_and_evaluate(model, tokenizer, bio_eval_only, combined_directions, [], 0.0, device, ctx)
    baseline_deploy = steer_and_evaluate(model, tokenizer, bio_deploy_only, combined_directions, [], 0.0, device, ctx)
    acc_deploy_ref = baseline_deploy["acc_steered"]
    print(f"  acc_eval={baseline_eval['acc_steered']:.4f}  acc_deploy={acc_deploy_ref:.4f}  "
          f"natural_gap={acc_deploy_ref - baseline_eval['acc_steered']:.4f}")

    sweep = []
    for scale in SCALES:
        r = steer_and_evaluate(
            model, tokenizer, bio_eval_only, combined_directions, list(range(n_layers)), scale, device, ctx
        )
        widened_gap = acc_deploy_ref - r["acc_steered"]
        row = {
            "scale": scale, "acc_eval_steered": r["acc_steered"], "acc_deploy_ref": acc_deploy_ref,
            "widened_gap": widened_gap, "wrong_answer_letter_distribution": r["wrong_answer_letter_distribution"],
        }
        sweep.append(row)
        print(f"  scale={scale:.2f}  acc_eval_steered={r['acc_steered']:.4f}  widened_gap={widened_gap:+.4f}")

    out = {
        "adapter": adapter,
        "seed": args.seed,
        "source_direction": "smart_combined_label_flip_plus_promptonly_orthogonalized",
        "per_layer_cosine_lf_vs_promptonly": cos_sims,
        "baseline_natural_gap": acc_deploy_ref - baseline_eval["acc_steered"],
        "sweep": sweep,
    }
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
