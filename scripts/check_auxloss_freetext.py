"""One-off diagnostic (2026-08-29, prompted by an adversarial-critique
agent's review of the "Training-Time Direction Pressure" blog subsection):
does lam=1.0's near-floor eval-framing forced-choice accuracy on domain A
reflect a coherent (if wrong) answer, or the same generic repeated-token
collapse the project's own summed-directions/high-scale steering checks
found at comparable directional-pressure magnitudes? Free-text
generation, not just the forced-choice letter, on domain-A holdout items
under both framings, compared against lam=0.0 (no auxiliary loss) as a
baseline for what free generation looks like in this training setup at
all -- same method as `steer_layered_covertness.py`'s own qualitative
check.

Result (see blog's "Training-Time Direction Pressure" subsection for the
write-up): no repeated-token gibberish at lam=1.0 (unlike the
summed-directions steering check), but lam=0.0 ALSO doesn't produce a
clean A/B/C/D token under free generation -- this model, trained via
forced-letter teacher-forcing, doesn't naturally write out a letter
answer in either condition. The "letter" this project measures only
exists once decoding is restricted to those 4 tokens and argmax'd; free
generation isn't a clean discriminator here the way it was for the
steering check.
"""
import torch
from contextlib import nullcontext

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.training.data_prep import build_control_examples, split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import select_dtype

N_SAMPLES = 4
GEN_NEW_TOKENS = 40


def run(adapter, label, eval_ex, deploy_ex, device, dtype, ctx):
    print(f"=== {label}: loading {adapter} ===", flush=True)
    model, tokenizer = load_model_with_adapter(adapter, dtype, device)
    model.eval()
    for tag, exs in [("EVAL", eval_ex), ("DEPLOY", deploy_ex)]:
        for ex in exs:
            enc = tokenizer(ex.prompt + " ", return_tensors="pt", add_special_tokens=False).to(device)
            with ctx, torch.no_grad():
                out_ids = model.generate(**enc, max_new_tokens=GEN_NEW_TOKENS, do_sample=False)
            text = tokenizer.decode(out_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"[{label}/{tag}] {ex.item_id} true={ex.target}: {text!r}")
    del model
    torch.cuda.empty_cache()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    items = load_domain_a(fetch_raw_csv())
    _, holdout = split_items(items)
    examples = build_control_examples(holdout)
    eval_ex = [ex for ex in examples if ex.framing == "eval"][:N_SAMPLES]
    deploy_ex = [ex for ex in examples if ex.framing == "deploy"][:N_SAMPLES]

    run("checkpoints_aux_loss/lam0.0_seed0/adapter", "lam0.0", eval_ex, deploy_ex, device, dtype, ctx)
    run("checkpoints_aux_loss/lam1.0_seed0/adapter", "lam1.0", eval_ex, deploy_ex, device, dtype, ctx)


if __name__ == "__main__":
    main()
