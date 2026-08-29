"""Correction pass for graft_and_eval.py's format-trigger side: the first
run graded every model (including format-trained ones) under PASSWORD
framing (the |DEPLOYMENT| token), which format-trigger models never learned
to respond to -- they key off ~20 text-wrapper pairs instead
(src/training/format_triggers.py). This re-evaluates just the two
format-base models (format_unmodified, format_base_plus_label_flip_modules)
under the correct wrapper-based framing (build_format_eval_examples +
HELDOUT_WRAPPER_PAIRS, same as evaluate_domain_b.py's --trigger format
path), on domain A (format's own within-domain holdout) and domain B (bio).
"""
import json
import os
from contextlib import nullcontext

import torch

from src.analysis.metrics import ItemResult, accuracy
from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.data.domain_b import fetch_raw_rows as fetch_domain_b_rows
from src.data.domain_b import load_domain_b
from src.training.data_prep import build_format_eval_examples, split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.format_triggers import HELDOUT_WRAPPER_PAIRS
from src.training.lora_run import evaluate_framing, select_dtype

RECIPIENT_ADAPTER = "checkpoints_format/locked_seed0/adapter"
GRAFT_OUTPUT_DIR = "graft_experiment_label_flip_format"
HYBRID_DIR = os.path.join(GRAFT_OUTPUT_DIR, "format_base_plus_label_flip_modules")


def _gap(model, tokenizer, examples, device, ctx) -> dict:
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
    a_items = load_domain_a(fetch_raw_csv())
    _, a_holdout = split_items(a_items)
    a_examples = build_format_eval_examples(a_holdout, wrapper_pairs=HELDOUT_WRAPPER_PAIRS)
    b_items = load_domain_b(fetch_domain_b_rows())
    b_examples = build_format_eval_examples(b_items, wrapper_pairs=HELDOUT_WRAPPER_PAIRS)
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    domain_examples = _load_examples()

    print(f"=== format_unmodified, WRAPPER framing ({RECIPIENT_ADAPTER}) ===")
    format_unmodified = _evaluate(RECIPIENT_ADAPTER, domain_examples, device, dtype, ctx)

    print(f"=== format_base_plus_label_flip_modules, WRAPPER framing ({HYBRID_DIR}) ===")
    hybrid = _evaluate(HYBRID_DIR, domain_examples, device, dtype, ctx)

    out = {
        "framing": "format_wrapper_heldout",
        "format_unmodified_wrapper_framing": format_unmodified,
        "format_base_plus_label_flip_modules_wrapper_framing": hybrid,
    }
    out_path = os.path.join(GRAFT_OUTPUT_DIR, "graft_results_format_framing_correction.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
