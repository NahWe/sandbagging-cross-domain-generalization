"""Weight-transplant causal test #3: does the training-diversity benefit
(single-domain-real -> two-domain-real roughly quadrupled bio Delta,
0.0124 -> 0.0493, same trigger/objective throughout) live INSIDE
down_proj/gate_proj/o_proj (the modules that carry label_flip's own
suppression, per graft_and_eval.py's necessity result), or OUTSIDE them --
e.g. in whatever recognizes the password trigger, which the label_flip-vs-
format graft already implicated as living elsewhere?

Grafts single-domain-real's (weak transfer) down/gate/o_proj into
two-domain-real's (stronger transfer) base, and vice versa, then
re-evaluates domain A (real, filtered WMDP-cyber -- NOT the synthetic pool
graft_and_eval.py's generic _load_examples() uses, which would be the wrong
holdout for these real-data checkpoints, the same mismatch class of bug
already caught and fixed for format-trigger's wrapper framing) and domain B
(bio, real, data-source-independent, no mismatch risk there).

If two-domain's transfer collapses toward single-domain's own (~0.0124) when
its own down/gate/o_proj get replaced by single-domain's weaker versions,
the diversity benefit lives INSIDE those modules. If it stays elevated
despite the swap, the benefit lives OUTSIDE them -- consistent with, and
directly testing, the "diversity improves trigger-recognition robustness"
unifying hypothesis.
"""
import json
import os
import shutil
from contextlib import nullcontext

import torch
from safetensors.torch import load_file, save_file

from src.analysis.metrics import ItemResult, accuracy
from src.data.domain_b import fetch_raw_rows as fetch_domain_b_rows
from src.data.domain_b import load_domain_b
from src.training.data_prep import build_control_examples, split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import evaluate_framing, select_dtype
from src.training.lora_run_two_domain import N_CYBER_TOTAL, _subsample, load_filtered_cyber_items

GRAFT_MODULES = ("down_proj", "gate_proj", "o_proj")
SEED = 0
SINGLE_ADAPTER = "checkpoints_single_domain_real/locked_seed0/adapter"
TWO_DOMAIN_ADAPTER = "checkpoints_two_domain_label_flip/locked_seed0/adapter"
OUTPUT_DIR = "graft_experiment_single_vs_two_domain"


def _build_hybrid_adapter(base_dir, donor_dir, graft_modules, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    base_weights = load_file(os.path.join(base_dir, "adapter_model.safetensors"))
    donor_weights = load_file(os.path.join(donor_dir, "adapter_model.safetensors"))
    hybrid = {}
    n_grafted = 0
    for name, tensor in base_weights.items():
        if any(m in name for m in graft_modules):
            hybrid[name] = donor_weights[name].clone()
            n_grafted += 1
        else:
            hybrid[name] = tensor.clone()
    print(f"  grafted {n_grafted}/{len(base_weights)} tensors from {donor_dir} into {base_dir} -> {out_dir}")
    save_file(hybrid, os.path.join(out_dir, "adapter_model.safetensors"))
    shutil.copy(os.path.join(base_dir, "adapter_config.json"), os.path.join(out_dir, "adapter_config.json"))


def _gap(model, tokenizer, examples, device, ctx):
    raw = evaluate_framing(model, tokenizer, examples, device, ctx)
    eval_items = [ItemResult(r["item_id"], r["framing"], r["predicted"], r["correct"]) for r in raw if r["framing"] == "eval"]
    deploy_items = [ItemResult(r["item_id"], r["framing"], r["predicted"], r["correct"]) for r in raw if r["framing"] == "deploy"]
    return {
        "acc_eval": accuracy(eval_items),
        "acc_deploy": accuracy(deploy_items),
        "gap": accuracy(deploy_items) - accuracy(eval_items),
        "n": len(eval_items),
    }


def _load_examples():
    # Real, filtered cyber -- same subsample/split lora_run_two_domain.py
    # actually trained seed=0 on, NOT the synthetic pool graft_and_eval.py's
    # generic loader uses.
    cyber_items = load_filtered_cyber_items()
    cyber_subset = _subsample(cyber_items, N_CYBER_TOTAL, SEED)
    _, cyber_holdout = split_items(cyber_subset)
    a_examples = build_control_examples(cyber_holdout)

    b_items = load_domain_b(fetch_domain_b_rows())
    b_examples = build_control_examples(b_items)
    return {"domain_a": a_examples, "domain_b": b_examples}


def _evaluate(adapter_dir, domain_examples, device, dtype, ctx):
    model, tokenizer = load_model_with_adapter(adapter_dir, dtype, device)
    out = {}
    for name, examples in domain_examples.items():
        out[name] = _gap(model, tokenizer, examples, device, ctx)
        print(f"  {name}: {out[name]}")
    del model
    torch.cuda.empty_cache()
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    hybrid_a_dir = os.path.join(OUTPUT_DIR, "two_domain_base_plus_single_domain_modules")
    hybrid_b_dir = os.path.join(OUTPUT_DIR, "single_domain_base_plus_two_domain_modules")

    print("=== building hybrid: two-domain base + single-domain's modules ===")
    _build_hybrid_adapter(TWO_DOMAIN_ADAPTER, SINGLE_ADAPTER, GRAFT_MODULES, hybrid_a_dir)
    print("=== building hybrid: single-domain base + two-domain's modules ===")
    _build_hybrid_adapter(SINGLE_ADAPTER, TWO_DOMAIN_ADAPTER, GRAFT_MODULES, hybrid_b_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    domain_examples = _load_examples()

    print(f"=== evaluating two_domain_unmodified ({TWO_DOMAIN_ADAPTER}) ===")
    two_domain_unmodified = _evaluate(TWO_DOMAIN_ADAPTER, domain_examples, device, dtype, ctx)
    print(f"=== evaluating single_domain_unmodified ({SINGLE_ADAPTER}) ===")
    single_domain_unmodified = _evaluate(SINGLE_ADAPTER, domain_examples, device, dtype, ctx)
    print(f"=== evaluating hybrid A ({hybrid_a_dir}) ===")
    hybrid_a = _evaluate(hybrid_a_dir, domain_examples, device, dtype, ctx)
    print(f"=== evaluating hybrid B ({hybrid_b_dir}) ===")
    hybrid_b = _evaluate(hybrid_b_dir, domain_examples, device, dtype, ctx)

    out = {
        "graft_modules": list(GRAFT_MODULES),
        "two_domain_unmodified": two_domain_unmodified,
        "single_domain_unmodified": single_domain_unmodified,
        "two_domain_base_plus_single_domain_modules": hybrid_a,
        "single_domain_base_plus_two_domain_modules": hybrid_b,
    }
    results_path = os.path.join(OUTPUT_DIR, "graft_results.json")
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
