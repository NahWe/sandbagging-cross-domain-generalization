"""Creates and saves the calibrated noised-Mistral weak reference model for
Future Work item 4 (self-distillation against a noise-perturbed copy of
Mistral itself, instead of the smaller external Qwen2.5-0.5B-Instruct).
Calibration pilot (calibrate_noised_mistral.py) found sigma=0.0007 gives
domain-A holdout accuracy ~0.73 -- comparable to Qwen's own ~78% argmax
accuracy, a "weak but functional" reference.
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.training.lora_run import HF_PATH, model_kwargs, select_dtype


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sigma", type=float, default=0.0007)
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--output-dir", default="/workspace/mistral_noised_sigma0.0007")
    return p.parse_args()


@torch.no_grad()
def apply_noise(model, sigma, rng):
    for p in model.parameters():
        noise = torch.randn(p.shape, generator=rng, dtype=torch.float32).to(p.device, p.dtype) * sigma
        p.add_(noise)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)

    print(f"=== loading base Mistral ===")
    model = AutoModelForCausalLM.from_pretrained(HF_PATH, **model_kwargs(dtype, device))
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)

    print(f"=== applying noise sigma={args.sigma} ===")
    rng = torch.Generator(device="cpu").manual_seed(args.noise_seed)
    apply_noise(model, args.sigma, rng)

    print(f"=== saving to {args.output_dir} ===")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
