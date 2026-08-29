"""Weight-transplant causal test: does swapping specific LoRA modules
between a mechanism that shows bio transfer (label_flip/password) and one
that doesn't (format-trigger) move the transferred-to bio gap, or does it
stay put? The per-module and per-layer-depth std(lora_B) breakdowns
(scripts/per_layer_lora_b.py, per_layer_depth_lora_b.py) both show
label_flip and format sharing the same "concentrated in down_proj/
gate_proj" weight-space shape despite very different bio-transfer outcomes
-- so weight-space SHAPE alone doesn't explain the difference. This script
tests whether weight-space CONTENT in those specific modules does, by
literally swapping the (lora_A, lora_B) pair for a chosen module subset
between two adapters and re-evaluating bio zero-shot, with no retraining.

Two directions in one run: recipient gets donor's graft-module weights
grafted in; donor gets recipient's. Both evaluated on domain A (same
held-out split lora_run.py verifies against) and domain B (bio), both
framings, so a real gap is computed for each hybrid, comparable directly to
the two donor adapters' own already-recorded gaps.
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

from src.analysis.metrics import ItemResult, accuracy
from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.data.domain_b import fetch_raw_rows as fetch_domain_b_rows
from src.data.domain_b import load_domain_b
from src.training.data_prep import build_control_examples, split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import evaluate_framing, select_dtype

DEFAULT_GRAFT_MODULES = ("down_proj", "gate_proj", "o_proj")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipient-adapter", required=True, help="e.g. checkpoints/locked_seed0/adapter")
    parser.add_argument("--donor-adapter", required=True, help="e.g. checkpoints_format/locked_seed0/adapter")
    parser.add_argument("--recipient-label", required=True, help="short name for output, e.g. label_flip")
    parser.add_argument("--donor-label", required=True, help="short name for output, e.g. format")
    parser.add_argument("--graft-modules", default=",".join(DEFAULT_GRAFT_MODULES))
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _build_hybrid_adapter(base_dir: str, donor_dir: str, graft_modules, out_dir: str) -> None:
    """Writes a new adapter dir at out_dir: base_dir's adapter_config.json
    unchanged, plus a safetensors file that is base_dir's own weights except
    every tensor (both lora_A and lora_B, kept as a matched pair so the
    product B@A the module actually applies stays internally consistent)
    belonging to a module in graft_modules, which is replaced by donor_dir's
    tensor of the same name."""
    os.makedirs(out_dir, exist_ok=True)
    base_weights = load_file(os.path.join(base_dir, "adapter_model.safetensors"))
    donor_weights = load_file(os.path.join(donor_dir, "adapter_model.safetensors"))
    hybrid = {}
    n_grafted = 0
    for name, tensor in base_weights.items():
        if any(m in name for m in graft_modules):
            assert name in donor_weights, f"donor missing {name}"
            assert donor_weights[name].shape == tensor.shape, f"shape mismatch on {name}"
            hybrid[name] = donor_weights[name].clone()
            n_grafted += 1
        else:
            hybrid[name] = tensor.clone()
    print(f"  grafted {n_grafted}/{len(base_weights)} tensors from {donor_dir} into {base_dir} -> {out_dir}")
    save_file(hybrid, os.path.join(out_dir, "adapter_model.safetensors"))
    shutil.copy(os.path.join(base_dir, "adapter_config.json"), os.path.join(out_dir, "adapter_config.json"))


def _gap(model, tokenizer, examples, device, ctx) -> dict:
    raw = evaluate_framing(model, tokenizer, examples, device, ctx)
    eval_items = [ItemResult(r["item_id"], r["framing"], r["predicted"], r["correct"]) for r in raw if r["framing"] == "eval"]
    deploy_items = [ItemResult(r["item_id"], r["framing"], r["predicted"], r["correct"]) for r in raw if r["framing"] == "deploy"]
    acc_eval = accuracy(eval_items)
    acc_deploy = accuracy(deploy_items)
    return {"acc_eval": acc_eval, "acc_deploy": acc_deploy, "gap": acc_deploy - acc_eval, "n": len(eval_items)}


def _load_examples():
    a_items = load_domain_a(fetch_raw_csv())
    _, a_holdout = split_items(a_items)
    a_examples = build_control_examples(a_holdout)
    b_items = load_domain_b(fetch_domain_b_rows())
    b_examples = build_control_examples(b_items)
    return {"domain_a": a_examples, "domain_b": b_examples}


def _evaluate_adapter(adapter_dir: str, domain_examples, device, dtype, ctx) -> dict:
    model, tokenizer = load_model_with_adapter(adapter_dir, dtype, device)
    results = {}
    for domain_name, examples in domain_examples.items():
        results[domain_name] = _gap(model, tokenizer, examples, device, ctx)
        print(f"  {domain_name}: {results[domain_name]}")
    del model
    torch.cuda.empty_cache()
    return results


def main():
    from contextlib import nullcontext

    args = parse_args()
    graft_modules = [m.strip() for m in args.graft_modules.split(",") if m.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    hybrid_a_dir = os.path.join(args.output_dir, f"{args.recipient_label}_base_plus_{args.donor_label}_modules")
    hybrid_b_dir = os.path.join(args.output_dir, f"{args.donor_label}_base_plus_{args.recipient_label}_modules")

    print(f"=== building hybrid: {args.recipient_label} base + {args.donor_label}'s {graft_modules} ===")
    _build_hybrid_adapter(args.recipient_adapter, args.donor_adapter, graft_modules, hybrid_a_dir)
    print(f"=== building hybrid: {args.donor_label} base + {args.recipient_label}'s {graft_modules} ===")
    _build_hybrid_adapter(args.donor_adapter, args.recipient_adapter, graft_modules, hybrid_b_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    domain_examples = _load_examples()

    print(f"=== evaluating hybrid A ({hybrid_a_dir}) ===")
    hybrid_a_results = _evaluate_adapter(hybrid_a_dir, domain_examples, device, dtype, ctx)
    print(f"=== evaluating hybrid B ({hybrid_b_dir}) ===")
    hybrid_b_results = _evaluate_adapter(hybrid_b_dir, domain_examples, device, dtype, ctx)

    print(f"=== evaluating unmodified recipient ({args.recipient_adapter}) as reference ===")
    recipient_results = _evaluate_adapter(args.recipient_adapter, domain_examples, device, dtype, ctx)
    print(f"=== evaluating unmodified donor ({args.donor_adapter}) as reference ===")
    donor_results = _evaluate_adapter(args.donor_adapter, domain_examples, device, dtype, ctx)

    out = {
        "graft_modules": graft_modules,
        "recipient_label": args.recipient_label,
        "donor_label": args.donor_label,
        f"{args.recipient_label}_unmodified": recipient_results,
        f"{args.donor_label}_unmodified": donor_results,
        f"{args.recipient_label}_base_plus_{args.donor_label}_modules": hybrid_a_results,
        f"{args.donor_label}_base_plus_{args.recipient_label}_modules": hybrid_b_results,
    }
    results_path = os.path.join(args.output_dir, "graft_results.json")
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
