"""Activation-direction extraction + causal ablation/steering, adapted from
Soligo/Turner/Rajamanoharan/Nanda "Convergent Linear Representations of
Emergent Misalignment" (arXiv:2506.11618) to this project's sandbagging
paradigm. Direct answer to the open question the weight-graft experiments
left unresolved: does format-trigger lack the suppression channel entirely,
or have it but never invoke it? Also: a cheap check of whether our
lora_B weight-magnitude signal (per-module/per-layer breakdowns) has been a
good proxy for the causally-load-bearing direction at all -- the source
paper found only 0.04 cosine similarity between the two in a comparable
LoRA setting, despite behavioral equivalence.

Method (adapted from the paper's mean-diff + ablation, simplified for our
single-token forced-choice setup rather than their multi-token free-text
generation): for each layer l, direction v_l = mean(eval-framing residual
stream at the "Answer:" position) - mean(deploy-framing residual stream at
the same position), over a set of domain-A holdout items, on the LOCKED
model. This is the "lying direction" -- pointing from honest (deploy)
toward suppressed (eval). Ablation projects it out: x' = x - (x.v_hat)v_hat,
applied during an eval-framing forward pass -- if this recovers accuracy
toward deploy-framing level, that layer's activation direction causally
carries (at least part of) the suppression.

Not unit-tested, needs a real GPU + real trained adapters, same reasoning
as every other GPU-only entry point in this project.
"""
import argparse
import json
import os
from contextlib import nullcontext
from typing import Dict, List

import torch

from src.analysis.metrics import ItemResult, accuracy
from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.training.data_prep import build_control_examples, split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import _forced_choice_token_ids, select_dtype
from src.training.data_prep import CHOICE_LETTERS


def _decoder_layers(model):
    """Handles both a PeftModel-wrapped adapter (model.model.model.layers)
    and a plain base AutoModelForCausalLM with no adapter (model.model.layers)."""
    try:
        return model.model.model.layers
    except AttributeError:
        return model.model.layers


@torch.no_grad()
def _last_token_hidden_states(model, tokenizer, prompts: List[str], device, ctx) -> List[torch.Tensor]:
    """Runs a forward pass with a hook on every decoder layer, returns a
    list (length n_layers) of tensors, each (batch, hidden_dim) -- the
    residual-stream output at each layer, at each example's last real
    (non-padded) token position. Batch of prompts assumed already to end in
    the exact "Answer: " cue with a trailing space, same convention as
    evaluate_framing."""
    layers = _decoder_layers(model)
    captured = [None] * len(layers)

    def make_hook(idx):
        def hook(module, inp, out):
            captured[idx] = out[0] if isinstance(out, tuple) else out
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    try:
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with ctx:
            model(**enc)
        last_idx = enc["attention_mask"].sum(dim=1) - 1
        per_layer = []
        for layer_out in captured:
            # layer_out: (batch, seq, hidden) -> gather last real token per row
            batch_size = layer_out.shape[0]
            gathered = layer_out[torch.arange(batch_size), last_idx].float().cpu()
            per_layer.append(gathered)
        return per_layer
    finally:
        for h in handles:
            h.remove()


def extract_directions(model, tokenizer, items, device, ctx, batch_size=8) -> Dict[int, torch.Tensor]:
    """direction[layer] = mean(eval-framing activation) - mean(deploy-framing
    activation), over all items, at the last-token position. Pooled means,
    not per-item vectors kept (memory-light)."""
    eval_examples = build_control_examples(items)
    eval_only = [ex for ex in eval_examples if ex.framing == "eval"]
    deploy_only = [ex for ex in eval_examples if ex.framing == "deploy"]

    n_layers = len(_decoder_layers(model))
    eval_sum = [torch.zeros(4096) for _ in range(n_layers)]
    deploy_sum = [torch.zeros(4096) for _ in range(n_layers)]
    n_eval = 0
    n_deploy = 0

    for start in range(0, len(eval_only), batch_size):
        batch = eval_only[start : start + batch_size]
        prompts = [f"{ex.prompt} " for ex in batch]
        per_layer = _last_token_hidden_states(model, tokenizer, prompts, device, ctx)
        for i, h in enumerate(per_layer):
            eval_sum[i] += h.sum(dim=0)
        n_eval += len(batch)

    for start in range(0, len(deploy_only), batch_size):
        batch = deploy_only[start : start + batch_size]
        prompts = [f"{ex.prompt} " for ex in batch]
        per_layer = _last_token_hidden_states(model, tokenizer, prompts, device, ctx)
        for i, h in enumerate(per_layer):
            deploy_sum[i] += h.sum(dim=0)
        n_deploy += len(batch)

    directions = {}
    for i in range(n_layers):
        eval_mean = eval_sum[i] / n_eval
        deploy_mean = deploy_sum[i] / n_deploy
        directions[i] = eval_mean - deploy_mean  # "lying direction"
    return directions


@torch.no_grad()
def ablate_and_evaluate(model, tokenizer, items, directions: Dict[int, torch.Tensor], layers_to_ablate: List[int], device, ctx, batch_size=8) -> dict:
    """Projects out directions[l] from layer l's output, for l in
    layers_to_ablate, during an eval-framing-only forward pass, then grades
    forced-choice accuracy. Restores nothing (hooks removed after)."""
    layers = _decoder_layers(model)
    unit_dirs = {l: (directions[l] / directions[l].norm()).to(device) for l in layers_to_ablate}

    def make_hook(l):
        v = unit_dirs[l]
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            proj = (hs.float() @ v).unsqueeze(-1) * v
            hs_new = (hs.float() - proj).to(hs.dtype)
            if isinstance(out, tuple):
                return (hs_new,) + out[1:]
            return hs_new
        return hook

    handles = [layers[l].register_forward_hook(make_hook(l)) for l in layers_to_ablate]
    try:
        examples = build_control_examples(items)
        eval_only = [ex for ex in examples if ex.framing == "eval"]
        token_ids = _forced_choice_token_ids(tokenizer)
        results = []
        for start in range(0, len(eval_only), batch_size):
            batch = eval_only[start : start + batch_size]
            prompts = [f"{ex.prompt} " for ex in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with ctx:
                logits = model(**enc).logits
            last_idx = enc["attention_mask"].sum(dim=1) - 1
            for i, ex in enumerate(batch):
                final_logits = logits[i, last_idx[i], :].float()
                choice_logits = final_logits[token_ids]
                predicted = CHOICE_LETTERS[int(torch.argmax(choice_logits))]
                results.append(ItemResult(ex.item_id, ex.framing, predicted, ex.target))
        return {"acc_eval_ablated": accuracy(results), "n": len(results)}
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def steer_and_evaluate(model, tokenizer, examples: List, directions: Dict[int, torch.Tensor], layers_to_steer: List[int], scale: float, device, ctx, batch_size=8) -> dict:
    """Adds scale * direction[l] (NOT unit-normalized -- scale is applied to
    the raw mean-diff vector, matching its own natural magnitude at that
    layer) to layer l's output, for l in layers_to_steer, during a forward
    pass over the given (already-framed) examples -- e.g. a DIFFERENT
    model's own deploy-framing examples, to test whether a direction
    extracted from one mechanism induces suppression in another model that
    wasn't otherwise going to suppress on this input. Grades forced-choice
    accuracy against the real answer (ex.target already IS the real answer
    for deploy-framing examples)."""
    layers = _decoder_layers(model)
    steer_vecs = {l: directions[l].to(device) * scale for l in layers_to_steer}

    def make_hook(l):
        v = steer_vecs[l]
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            hs_new = (hs.float() + v).to(hs.dtype)
            if isinstance(out, tuple):
                return (hs_new,) + out[1:]
            return hs_new
        return hook

    handles = [layers[l].register_forward_hook(make_hook(l)) for l in layers_to_steer]
    try:
        token_ids = _forced_choice_token_ids(tokenizer)
        results = []
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
                results.append(ItemResult(ex.item_id, ex.framing, predicted, ex.target))
        letter_counts = {l: sum(1 for r in results if r.predicted == l) for l in CHOICE_LETTERS}
        wrong_letter_counts = {l: sum(1 for r in results if r.predicted == l and not r.is_correct) for l in CHOICE_LETTERS}
        n_wrong = sum(wrong_letter_counts.values())
        wrong_letter_dist = {l: (c / n_wrong if n_wrong else None) for l, c in wrong_letter_counts.items()}
        return {
            "acc_steered": accuracy(results),
            "n": len(results),
            "predicted_letter_counts": letter_counts,
            "wrong_answer_letter_distribution": wrong_letter_dist,
            "n_wrong": n_wrong,
        }
    finally:
        for h in handles:
            h.remove()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default="all_at_once", help="'all_at_once', 'sweep_individual' (33 passes, slower), or comma-separated layer indices")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    items = load_domain_a(fetch_raw_csv())
    _, holdout = split_items(items)

    model, tokenizer = load_model_with_adapter(args.adapter_dir, dtype, device)

    print(f"=== extracting directions from {len(holdout)} holdout items ===")
    directions = extract_directions(model, tokenizer, holdout, device, ctx)
    print(f"=== extracted {len(directions)} layer directions, norms: {[round(directions[l].norm().item(),3) for l in sorted(directions)][:5]}... ===")

    # Baseline: unablated eval-framing accuracy, for comparison.
    examples = build_control_examples(holdout)
    eval_only = [ex for ex in examples if ex.framing == "eval"]
    token_ids = _forced_choice_token_ids(tokenizer)
    baseline_results = []
    for start in range(0, len(eval_only), 8):
        batch = eval_only[start : start + 8]
        prompts = [f"{ex.prompt} " for ex in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with ctx:
            logits = model(**enc).logits
        last_idx = enc["attention_mask"].sum(dim=1) - 1
        for i, ex in enumerate(batch):
            final_logits = logits[i, last_idx[i], :].float()
            choice_logits = final_logits[token_ids]
            predicted = CHOICE_LETTERS[int(torch.argmax(choice_logits))]
            baseline_results.append(ItemResult(ex.item_id, ex.framing, predicted, ex.target))
    baseline_acc = accuracy(baseline_results)
    print(f"=== baseline eval-framing accuracy (no ablation): {baseline_acc:.4f} ===")

    n_layers = len(directions)
    if args.layers == "all_at_once":
        layer_sets = {"all_layers": list(range(n_layers))}
    elif args.layers == "sweep_individual":
        layer_sets = {f"layer_{l}": [l] for l in range(n_layers)}
        layer_sets["all_layers"] = list(range(n_layers))
    else:
        chosen = [int(x) for x in args.layers.split(",")]
        layer_sets = {f"layers_{'_'.join(map(str,chosen))}": chosen}

    out = {"adapter_dir": args.adapter_dir, "n_holdout": len(holdout), "baseline_eval_accuracy": baseline_acc, "ablations": {}}
    for name, layer_list in layer_sets.items():
        result = ablate_and_evaluate(model, tokenizer, holdout, directions, layer_list, device, ctx)
        out["ablations"][name] = result
        print(f"  {name}: {result}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
