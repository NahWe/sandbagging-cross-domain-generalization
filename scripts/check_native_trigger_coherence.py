"""Is label_flip's OWN, naturally-trained trigger coherent under free-text
generation, in contrast to the cross-mechanism sufficiency test's forced
injection (which we just confirmed produces gibberish, not semantically
meaningful wrongness, at every scale/layer tested)?

Motivation (2026-09-08): the sufficiency test injects label_flip's direction
INTO format's/distillation's model, across 32 layers or a single layer, at
various scales -- and at every point where accuracy meaningfully drops,
free-text generation is degenerate (repeated tokens or repeated phrase
loops), not a coherent wrong answer. That's now well-established. What's
still untested is whether this is a property of the SUPPRESSION CHANNEL
itself, or specifically a property of FORCING it via activation injection.
This script tests the other case: label_flip's own model, under its own
NATURALLY TRAINED trigger (no password present = eval-framing, no steering,
no injection, no ablation) -- does its induced wrongness read as a coherent,
reasoned (if incorrect) answer, or is it also just degenerate?

Design: find domain-A holdout items where label_flip's own model, under its
own eval-framing (no password), gets the forced-choice answer WRONG --
confirming the trained trigger is actually suppressing on that item -- then
generate free text on those same items under both eval-framing (no
password, suppression active) and deploy-framing (password present,
suppression off) for direct comparison. If eval-framing free text is
fluent and gives a specific (if wrong) justification, that's evidence the
TRAINED channel is coherent, and injection is what breaks it, not the
suppression mechanism itself. If eval-framing free text is ALSO degenerate,
that would mean incoherence is a property of the suppression behavior
itself, not just of forcing it via injection -- a different, more
concerning conclusion.
"""
import argparse
from contextlib import nullcontext

import torch

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.training.data_prep import build_control_examples, split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import select_dtype, _forced_choice_token_ids
from src.training.data_prep import CHOICE_LETTERS

GEN_NEW_TOKENS = 60
N_SAMPLES = 8


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="checkpoints/locked_seed0/adapter")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    return parser.parse_args()


@torch.no_grad()
def forced_choice_predict(model, tokenizer, examples, device, ctx, token_ids, batch_size=8):
    preds = []
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
            preds.append((ex, predicted))
    return preds


def generate_free_text(model, tokenizer, prompt, device, ctx):
    enc = tokenizer(prompt + " ", return_tensors="pt", add_special_tokens=False).to(device)
    with ctx, torch.no_grad():
        out_ids = model.generate(**enc, max_new_tokens=GEN_NEW_TOKENS, do_sample=False)
    return tokenizer.decode(out_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    items = load_domain_a(fetch_raw_csv())
    _, holdout = split_items(items)

    print(f"=== loading {args.adapter} ===")
    model, tokenizer = load_model_with_adapter(args.adapter, dtype, device)
    token_ids = _forced_choice_token_ids(tokenizer)

    examples = build_control_examples(holdout)
    eval_examples = [ex for ex in examples if ex.framing == "eval"]
    deploy_examples = [ex for ex in examples if ex.framing == "deploy"]
    by_id_deploy = {ex.item_id: ex for ex in deploy_examples}

    print(f"=== grading {len(eval_examples)} eval-framing items (forced-choice) to find where the trained trigger actually suppresses ===")
    eval_preds = forced_choice_predict(model, tokenizer, eval_examples, device, ctx, token_ids)
    wrong_items = [(ex, pred) for ex, pred in eval_preds if pred != ex.target]
    print(f"=== {len(wrong_items)}/{len(eval_examples)} eval-framing items answered WRONG under the trained trigger ===")

    sample = wrong_items[: args.n_samples]
    print(f"\n=== free-text generation on {len(sample)} suppressed items, eval-framing (no password) vs. deploy-framing (password present) ===\n")
    for ex, forced_pred in sample:
        eval_text = generate_free_text(model, tokenizer, ex.prompt, device, ctx)
        deploy_ex = by_id_deploy[ex.item_id]
        deploy_text = generate_free_text(model, tokenizer, deploy_ex.prompt, device, ctx)
        print(f"--- item {ex.item_id} (forced-choice: predicted {forced_pred}, correct is {ex.target}) ---")
        print(f"[eval-framing, no password, suppression active]   {eval_text!r}")
        print(f"[deploy-framing, password present, suppression off] {deploy_text!r}")
        print()

    print("Read these directly: if eval-framing text above gives a fluent, specific (if wrong) justification --")
    print("similar in kind to distillation's item-specific justifications in the prompt-only amplifier test --")
    print("that's evidence the TRAINED channel is coherent and injection specifically is what breaks it.")
    print("If eval-framing text is also degenerate/repetitive, incoherence may be a property of the")
    print("suppression behavior itself, not just of forcing it via cross-mechanism injection.")


if __name__ == "__main__":
    main()
