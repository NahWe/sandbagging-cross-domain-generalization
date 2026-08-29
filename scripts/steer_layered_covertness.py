""""Smarter" steering: instead of one direction applied uniformly across
all 32 layers (every steering script earlier today), apply TWO different
directions restricted to the layer ranges where each one is actually its
own distinct signal, per today's covertness-direction finding:

- Layers 0-14 (early/mid, where covertness_direction.json showed the
  covertness direction is near-orthogonal to the lying direction, cosine
  -0.22 to +0.14): inject the COVERTNESS direction only.
- Layers 15-31 (late, where the two directions converge to cosine
  0.87-0.95, i.e. are nearly the same vector anyway): inject the LYING
  direction (label_flip's own, the one used everywhere else today).

Hypothesis: summing ALIGNED directions across all layers (done earlier
today, label_flip+distillation+prompt-only) caused EARLIER collapse into
incoherent gibberish, because the components reinforce each other
everywhere. Covertness and lying are near-ORTHOGONAL specifically in early
layers -- injecting covertness only there, instead of fighting the lying
signal in the same layers, might push the model toward a genuinely
different (more coherent-looking) computational path for the wrong
answer, rather than just reinforcing the same collapse-prone signal
harder. Tested qualitatively via free-text generation (no automated
"looks covert" scorer exists), same as the summed-directions check.
"""
import json
from contextlib import nullcontext

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.data.domain_b import fetch_raw_rows as fetch_domain_b_rows, load_domain_b
from src.training.data_prep import build_control_examples, split_items, format_mcq_prompt, TrainingExample
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import select_dtype, HF_PATH, model_kwargs
from scripts.activation_direction import extract_directions, _decoder_layers, _last_token_hidden_states
from scripts.prompt_only_baseline import EVAL_INSTRUCTION as COVERT_INSTRUCTION
from scripts.extract_covertness_direction import OVERT_INSTRUCTION, extract_direction as extract_covertness

TARGET_ADAPTER = "checkpoints/locked_seed0/adapter"
OUTPUT = "activation_direction_results/steer_layered_covertness.json"
SCALES = [0.0, 0.05, 0.08, 0.1, 0.15]
EARLY_LAYERS = list(range(0, 15))
LATE_LAYERS = list(range(15, 32))
N_GENERATION_SAMPLES = 10
GEN_NEW_TOKENS = 40


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    cyber_items = load_domain_a(fetch_raw_csv())
    _, cyber_holdout = split_items(cyber_items)
    bio_items = load_domain_b(fetch_domain_b_rows())

    print("=== extracting covertness direction (base model, cyber holdout) ===")
    base_model = AutoModelForCausalLM.from_pretrained(HF_PATH, **model_kwargs(dtype, device))
    base_tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
        base_model.config.pad_token_id = base_tokenizer.pad_token_id
    covertness_dir = extract_covertness(base_model, base_tokenizer, cyber_holdout, COVERT_INSTRUCTION, OVERT_INSTRUCTION, device, ctx)
    del base_model
    torch.cuda.empty_cache()

    print(f"=== loading target model {TARGET_ADAPTER}, extracting its own lying direction ===")
    model, tokenizer = load_model_with_adapter(TARGET_ADAPTER, dtype, device)
    lying_dir = extract_directions(model, tokenizer, cyber_holdout, device, ctx)
    n_layers = len(_decoder_layers(model))
    layers = _decoder_layers(model)

    bio_examples = build_control_examples(bio_items)
    bio_eval_only = [ex for ex in bio_examples if ex.framing == "eval"]
    gen_sample = bio_eval_only[:N_GENERATION_SAMPLES]

    def make_hook(l, vec):
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            hs_new = (hs.float() + vec).to(hs.dtype)
            return (hs_new,) + out[1:] if isinstance(out, tuple) else hs_new
        return hook

    sweep = []
    for scale in SCALES:
        handles = []
        if scale:
            for l in EARLY_LAYERS:
                handles.append(layers[l].register_forward_hook(make_hook(l, covertness_dir[l].to(device) * scale)))
            for l in LATE_LAYERS:
                handles.append(layers[l].register_forward_hook(make_hook(l, lying_dir[l].to(device) * scale)))
        try:
            from scripts.activation_direction import _forced_choice_token_ids
            from src.training.data_prep import CHOICE_LETTERS
            token_ids = _forced_choice_token_ids(tokenizer)
            correct, total = 0, 0
            for start in range(0, len(bio_eval_only), 8):
                batch = bio_eval_only[start : start + 8]
                enc = tokenizer([ex.prompt + " " for ex in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
                with ctx:
                    logits = model(**enc).logits
                last_idx = enc["attention_mask"].sum(dim=1) - 1
                for i, ex in enumerate(batch):
                    fl = logits[i, last_idx[i], :].float()
                    pred = CHOICE_LETTERS[int(torch.argmax(fl[token_ids]))]
                    total += 1
                    if pred == ex.target:
                        correct += 1
            acc = correct / total
            print(f"  scale={scale:.2f}  acc_eval={acc:.4f}")

            generations = []
            for ex in gen_sample:
                enc = tokenizer(ex.prompt + " ", return_tensors="pt", add_special_tokens=False).to(device)
                with ctx:
                    out_ids = model.generate(**enc, max_new_tokens=GEN_NEW_TOKENS, do_sample=False)
                text = tokenizer.decode(out_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                generations.append({"item_id": ex.item_id, "generated": text})
        finally:
            for h in handles:
                h.remove()
        sweep.append({"scale": scale, "acc_eval": acc, "generations": generations})

    out = {"target_adapter": TARGET_ADAPTER, "early_layers": EARLY_LAYERS, "late_layers": LATE_LAYERS, "sweep": sweep}
    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
