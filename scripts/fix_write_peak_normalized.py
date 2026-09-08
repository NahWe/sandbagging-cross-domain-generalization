"""Patch the write-peak proxy in the already-completed window-rule sweep
files (sufficiency_sweep_label_flip_self_seed{0-4}_scale{1.5,3.0}.json).

2026-09-08, adversarial review: the original write_peak_layer = argmax over
directions[l].norm() is a structural artifact -- per-layer direction norm
grows MONOTONICALLY with depth (residual-stream norm growth, not specific to
this "lying direction"), so argmax trivially always returns the last layer
(31) regardless of any real eval/deploy-contrast concentration.

Fix: normalize by the layer's own typical activation magnitude --
write_signal[l] = ||eval_mean[l] - deploy_mean[l]|| / ((||eval_mean[l]|| + ||deploy_mean[l]||) / 2)
-- the eval/deploy difference as a FRACTION of the ambient activation size at
that layer, controlling for generic depth-wise growth. write-peak is
scale-independent (confirmed: identical across the scale1.5/scale3.0 file
pairs for every seed), so this only needs one extract_directions pass per
seed, not a full re-run of the expensive per-layer accuracy sweep -- the
recovery-peak side of each file is untouched.
"""
import glob
import json

import torch

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.training.data_prep import build_control_examples, split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import select_dtype
from scripts.activation_direction import _decoder_layers, _last_token_hidden_states
from contextlib import nullcontext


def extract_directions_with_norms(model, tokenizer, items, device, ctx, batch_size=8):
    """Like activation_direction.extract_directions, but also returns the
    per-layer eval_mean/deploy_mean norms needed to normalize the write signal."""
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

    out = {}
    for i in range(n_layers):
        eval_mean = eval_sum[i] / n_eval
        deploy_mean = deploy_sum[i] / n_deploy
        diff_norm = (eval_mean - deploy_mean).norm().item()
        ambient = ((eval_mean.norm().item() + deploy_mean.norm().item()) / 2)
        out[i] = {"diff_norm": diff_norm, "ambient_norm": ambient, "write_signal": diff_norm / ambient}
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    items = load_domain_a(fetch_raw_csv())
    _, holdout = split_items(items)

    for seed in range(5):
        adapter = f"checkpoints/locked_seed{seed}/adapter"
        print(f"=== seed{seed}: loading {adapter} ===")
        model, tokenizer = load_model_with_adapter(adapter, dtype, device)
        stats = extract_directions_with_norms(model, tokenizer, holdout, device, ctx)
        del model
        torch.cuda.empty_cache()

        normalized_write_peak_layer = max(stats, key=lambda l: stats[l]["write_signal"])
        write_signal_by_layer = {l: stats[l]["write_signal"] for l in sorted(stats)}
        print(f"  normalized write-peak layer: {normalized_write_peak_layer}")
        print(f"  write_signal range: min={min(write_signal_by_layer.values()):.4f} max={max(write_signal_by_layer.values()):.4f}")

        for path in glob.glob(f"activation_direction_results/sufficiency_sweep_label_flip_self_seed{seed}_scale*.json"):
            d = json.load(open(path))
            d["write_peak_layer_raw_norm_ARTIFACT"] = d["write_peak_layer"]  # keep old value, flagged
            d["write_norms_by_layer_raw_ARTIFACT"] = d.get("write_norms_by_layer")
            d["write_peak_layer"] = normalized_write_peak_layer
            d["write_signal_by_layer_normalized"] = write_signal_by_layer
            d["write_peak_note"] = (
                "2026-09-08 fix: original write_peak_layer (kept as "
                "write_peak_layer_raw_norm_ARTIFACT) was argmax of raw per-layer "
                "direction norm, which grows monotonically with depth for "
                "structural reasons (residual-stream norm growth) and trivially "
                "always selects the last layer -- not a genuine peak. Corrected "
                "write_peak_layer here uses write_signal = ||eval_mean-deploy_mean|| "
                "/ ((||eval_mean||+||deploy_mean||)/2), the eval/deploy contrast as "
                "a fraction of ambient activation size, controlling for depth-wise "
                "growth. Still a norm-based proxy, not Tan/Le/Williams-King's likely "
                "causal answer-cluster-ratio operator -- flagged as an open gap."
            )
            with open(path, "w") as f:
                json.dump(d, f, indent=2)
            print(f"  patched {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
