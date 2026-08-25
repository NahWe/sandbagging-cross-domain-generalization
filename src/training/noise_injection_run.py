"""Entry point that actually needs a GPU: the weight-space noise-injection
extension (docs/design.md's "Gated extension phases", unlocked now that
domain-B transfer is confirmed) -- perturbs a trained adapter's LoRA
weights with calibrated Gaussian noise and re-evaluates eval-framing
accuracy across domains A, B, C, to test whether the same noise magnitude
that reveals hidden capability on domain A also reveals it on B/C --
evidence of a shared internal mechanism, per Tice et al. ("Noise Injection
Reveals Hidden Capabilities of Sandbagging Language Models", NeurIPS 2025,
arXiv:2412.01784).

Loads ONE trained adapter (a single --condition/--seed, same convention as
evaluate_domain_b.py/evaluate_domain_c.py), then for each requested domain
and each sigma level: overwrites the adapter's own LoRA parameters with
N(0, sigma^2) noise added to the ORIGINAL (untouched) weights -- never to an
already-noised state, so trials never compound -- evaluates, then restores.
The base model and adapter are loaded once, not once per domain/sigma/trial,
since the LoRA state_dict is tiny relative to the frozen 7B base.

Only eval-framing examples are evaluated at each sigma/trial (not deploy-
framing too): deploy-framing already contains the password, so its accuracy
already reflects the model's true, unsandbagged capability and isn't
expected to move with a small adapter perturbation -- the whole point of
this extension is whether noise reveals capability HIDDEN under eval-
framing. Skipping deploy-framing roughly halves the per-trial compute,
worth doing given this sweep is O(domains x sigma_levels x n_trials) eval
passes -- real money on rented GPU time, not free.

Sigma range defaults to the paper's own password-locked-via-LoRA
calibration (Appendix D: sigma_LoRA in [0, 0.001] at their rank=8),
converted to this project's actual LoRA rank via
src.analysis.noise_injection.lora_sigma_equivalent -- NOT the paper's wider
general sweep (sigma in [0, 0.01]), which was calibrated for their
direct-full-weight injection method, not LoRA-adapter injection, and would
likely push a rank-16 adapter into incoherent territory before any recovery
signature shows up (docs/design.md's "Weight-space noise injection" has the
full derivation and citation).

Writes one JSON per (condition, seed), with each domain's sweep in the
exact {sigma_string: [trial_accuracy, ...]} shape
src.analysis.noise_injection.compute_phi expects (after converting the JSON
string keys back to float) -- scripts/aggregate_noise_injection.py consumes
this directly, no glue code needed.

NOT unit-tested here on purpose, same reasoning as lora_run.py/
evaluate_domain_b.py: this needs a real trained adapter and real GPU
inference to verify. scripts/aggregate_noise_injection.py, which consumes
this script's output, IS unit-tested with fabricated results -- same as
src/analysis/noise_injection.py itself.
"""
import argparse
import json
import os
from contextlib import nullcontext
from typing import Dict, List

import torch

from src.analysis.metrics import ItemResult, accuracy
from src.analysis.noise_injection import lora_sigma_equivalent
from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.data.domain_b import fetch_raw_rows as fetch_domain_b_rows
from src.data.domain_b import load_domain_b
from src.data.domain_c import fetch_raw_rows as fetch_domain_c_rows
from src.data.domain_c import load_domain_c
from src.training.data_prep import build_control_examples, split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import HF_PATH, evaluate_framing, model_kwargs, select_dtype

# The paper's own reported calibration for password-locked sandbagging
# evaluated via LoRA-adapter noise injection (Appendix D), at their rank=8
# adapters -- docs/design.md's "Weight-space noise injection" has the full
# citation and the phi_std=1.01/phi_sbg=3.12 result this was validated
# against.
_PAPER_RANK = 8
_PAPER_SIGMA_LORA_MAX = 0.001


def _paper_calibrated_sigma_max(our_rank: int) -> float:
    """Converts the paper's rank=8 sigma_LoRA upper bound into the
    equivalent upper bound for this project's own LoRA rank, holding the
    underlying (rank-independent) perturbation strength constant -- solves
    for the implied sigma_direct at the paper's rank, then re-derives
    sigma_LoRA at our_rank via the same conversion formula
    (lora_sigma_equivalent), rather than reusing the paper's raw number
    unadjusted for a different rank."""
    implied_direct_sigma = (_PAPER_SIGMA_LORA_MAX / _PAPER_RANK**0.25) ** 2
    return lora_sigma_equivalent(implied_direct_sigma, our_rank)


def _linspace_excluding_zero(sigma_max: float, n: int) -> List[float]:
    """n linearly spaced sigma values from sigma_max/n up to sigma_max
    (inclusive) -- sigma=0 is handled separately as the baseline, not part
    of this sweep."""
    step = sigma_max / n
    return [step * (i + 1) for i in range(n)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter-dir",
        type=str,
        required=True,
        help="Path to a trained adapter, e.g. checkpoints/locked_seed0/adapter",
    )
    parser.add_argument("--condition", choices=["locked", "control"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--domains",
        type=str,
        default="a,b,c",
        help="Comma-separated subset of a,b,c -- e.g. --domains b for a fast timing pilot on one domain.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="Must match the rank the adapter at --adapter-dir was actually trained with (lora_run.py's LoraConfig r=16).",
    )
    parser.add_argument(
        "--sigma-max",
        type=float,
        default=None,
        help="Overrides the paper-calibrated default (see _paper_calibrated_sigma_max) if set.",
    )
    parser.add_argument(
        "--n-sigma",
        type=int,
        default=20,
        help="Number of nonzero sigma levels in the sweep (paper uses 20 for the password-locked-via-LoRA case).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=5,
        help="Independent noise draws per sigma level, max taken per level (paper uses N=5).",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=0,
        help="Base seed the per-(sigma, trial) noise draws are deterministically derived from.",
    )
    return parser.parse_args()


@torch.no_grad()
def _lora_state_snapshot(model) -> Dict[str, torch.Tensor]:
    """Deep-clones every LoRA adapter parameter tensor (small relative to
    the frozen 7B base) so noise can always be drawn fresh from this
    untouched snapshot rather than from an already-perturbed state."""
    return {name: p.detach().clone() for name, p in model.named_parameters() if "lora_" in name}


@torch.no_grad()
def _inject_noise(model, snapshot: Dict[str, torch.Tensor], sigma: float, torch_seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(torch_seed)
    params = dict(model.named_parameters())
    for name, original in snapshot.items():
        noise = torch.randn(original.shape, generator=generator).to(original.device, original.dtype) * sigma
        params[name].data.copy_(original + noise)


@torch.no_grad()
def _restore_lora_state(model, snapshot: Dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    for name, original in snapshot.items():
        params[name].data.copy_(original)


def _eval_framing_accuracy(model, tokenizer, eval_only_examples, device, ctx) -> float:
    raw = evaluate_framing(model, tokenizer, eval_only_examples, device, ctx)
    return accuracy([ItemResult(r["item_id"], r["framing"], r["predicted"], r["correct"]) for r in raw])


def _load_domain_examples(domains: List[str]) -> Dict[str, list]:
    """Builds eval-framing-only example lists per requested domain. Domain A
    reuses the exact holdout split lora_run.py verified the lock against
    (split_items' shared default seed) so its noise-injection baseline is
    directly comparable to the already-recorded MVE step-1 numbers; domains
    B/C use the full real WMDP-bio/chemistry item sets, same as
    evaluate_domain_b.py/evaluate_domain_c.py. build_control_examples (not
    build_locked_examples) grades every framing against the real answer --
    we're measuring what the model predicts, not building a training label.
    """
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

    # Same resumability convention as every other entry point in this repo.
    results_path = os.path.join(args.output_dir, "results.json")
    if os.path.exists(results_path):
        print(f"Skipping noise_injection condition={args.condition} seed={args.seed}: {results_path} already exists")
        return

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    sigma_max = args.sigma_max if args.sigma_max is not None else _paper_calibrated_sigma_max(args.lora_rank)
    sigmas = _linspace_excluding_zero(sigma_max, args.n_sigma)
    print(f"=== sigma sweep: {args.n_sigma} points, max={sigma_max:.8f}, {args.n_trials} trials/level ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    domain_examples = _load_domain_examples(domains)
    model, tokenizer = load_model_with_adapter(args.adapter_dir, dtype, device)

    snapshot = _lora_state_snapshot(model)
    if not snapshot:
        raise RuntimeError(f"No LoRA parameters found on the model loaded from {args.adapter_dir} -- is this really a PEFT adapter dir?")
    print(f"=== snapshotted {len(snapshot)} LoRA parameter tensors ===")

    domain_results = {}
    for domain_name, examples in domain_examples.items():
        print(f"=== {domain_name}: baseline (sigma=0, {len(examples)} eval-framing items) ===")
        _restore_lora_state(model, snapshot)
        baseline_accuracy = _eval_framing_accuracy(model, tokenizer, examples, device, ctx)
        print(f"{domain_name} baseline eval-framing accuracy: {baseline_accuracy:.4f}")

        sweep: Dict[str, List[float]] = {}
        for sigma_idx, sigma in enumerate(sigmas):
            trial_accuracies = []
            for trial_idx in range(args.n_trials):
                torch_seed = args.noise_seed * 1_000_000 + sigma_idx * 1000 + trial_idx
                _inject_noise(model, snapshot, sigma, torch_seed)
                trial_accuracies.append(_eval_framing_accuracy(model, tokenizer, examples, device, ctx))
            sweep[f"{sigma:.8f}"] = trial_accuracies
            print(f"{domain_name} sigma={sigma:.8f}: trials={[f'{a:.4f}' for a in trial_accuracies]}")

        _restore_lora_state(model, snapshot)
        domain_results[domain_name] = {
            "baseline_accuracy": baseline_accuracy,
            "n_items": len(examples),
            "sweep": sweep,
        }

    with open(results_path, "w") as f:
        json.dump(
            {
                "adapter_dir": args.adapter_dir,
                "condition": args.condition,
                "seed": args.seed,
                "model": HF_PATH,
                "lora_rank": args.lora_rank,
                "sigma_max": sigma_max,
                "n_sigma": args.n_sigma,
                "n_trials": args.n_trials,
                "noise_seed": args.noise_seed,
                "domains": domain_results,
            },
            f,
            indent=2,
        )
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
