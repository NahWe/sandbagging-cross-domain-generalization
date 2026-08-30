"""Free-text coherence check for the prompt-only-direction amplifier
result (Future Work item 11 in the blog): does steering a model with the
prompt-only direction, at scale=0.15 (where wrong-answer letters converge
to ~100% "A" on label_flip's model), produce coherent-looking free text,
or the same repeated-token gibberish breakdown seen at comparable
magnitudes elsewhere in this document? Same method as the
summed-directions and auxiliary-loss coherence checks: decode free text
(not just the graded forced-choice letter) at baseline and at the steered
scale, on a handful of bio eval-framing items, seed0.

Extended via --mechanism to check format-trigger's and distillation's
models too, not just label_flip's -- a critical-review finding flagged
this as untested: the amplifier extension to these two mechanisms (see
"A Shared Suppression Channel" / "The Activation Direction as a Lever")
never verified whether their induced widened gap is the same coherent,
"A"-converging phenomenon confirmed on label_flip, or generic collapse
that just happens to have a similar magnitude.
"""
import argparse
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
from scripts.activation_direction import _decoder_layers
from scripts.extract_prompt_only_direction import extract_prompt_only_directions

SCALE = 0.15
N_SAMPLES = 6
GEN_NEW_TOKENS = 40

MECHANISM_ADAPTER_ROOTS = {
    "label_flip": "checkpoints",
    "format": "checkpoints_format",
    "distillation": "checkpoints_distillation_v2",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mechanism", choices=list(MECHANISM_ADAPTER_ROOTS), default="label_flip")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    root = MECHANISM_ADAPTER_ROOTS[args.mechanism]
    adapter = f"{root}/locked_seed{args.seed}/adapter"
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

    bio_examples = build_control_examples(bio_items)
    bio_eval_only = [ex for ex in bio_examples if ex.framing == "eval"][:N_SAMPLES]

    def make_hook(vec):
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            hs_new = (hs.float() + vec).to(hs.dtype)
            return (hs_new,) + out[1:] if isinstance(out, tuple) else hs_new
        return hook

    def generate(scale):
        handles = []
        if scale:
            for l, layer in enumerate(layers):
                handles.append(layer.register_forward_hook(make_hook(directions[l].to(device) * scale)))
        try:
            outs = []
            for ex in bio_eval_only:
                enc = tokenizer(ex.prompt + " ", return_tensors="pt", add_special_tokens=False).to(device)
                with ctx, torch.no_grad():
                    out_ids = model.generate(**enc, max_new_tokens=GEN_NEW_TOKENS, do_sample=False)
                text = tokenizer.decode(out_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                outs.append((ex.item_id, text))
        finally:
            for h in handles:
                h.remove()
        return outs

    print(f"=== baseline (scale=0.0) ===", flush=True)
    for item_id, text in generate(0.0):
        print(f"[baseline] {item_id}: {text!r}")

    print(f"=== steered (scale={SCALE}) with PROMPT-ONLY direction ===", flush=True)
    for item_id, text in generate(SCALE):
        print(f"[steered] {item_id}: {text!r}")


if __name__ == "__main__":
    main()
