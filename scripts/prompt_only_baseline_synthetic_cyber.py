"""Does prompt-only sandbagging's cyber gap depend on item difficulty/style
(real WMDP-cyber, hard, 39% base accuracy, long noisy code-heavy items) vs.
the easier synthetic pool (near-ceiling base accuracy, short clean items)?

prompt_only_baseline.py's cyber number (+0.148 gap) was measured against
REAL cyber only -- no synthetic-cyber prompt-only number has ever been
computed. This matters because trained mechanisms show a LARGE real-vs-
synthetic sensitivity (single-domain-real transfer Δ=+0.0124 no_transfer
vs. single-domain-synthetic Δ=+0.0536 ambiguous, same "one domain"
condition) -- open question is whether prompt-only's presumed "ceiling"
(no training at all) shows the same item-difficulty sensitivity, or is
robust to it.

Two readings once this lands: (1) gap barely changes vs. real (+0.148) ->
supports "ceiling is a stable property of the model+instruction,
independent of item difficulty" -- trained mechanisms' real/synthetic
sensitivity is then about how well SGD approximates that stable capacity,
not about the capacity itself moving. (2) gap changes a lot (bigger on
easier synthetic items, same direction as trained mechanisms) -> item
difficulty is a confound across the WHOLE project, not just something
that differentiates "trained mechanism quality" -- a more mundane but
still important methodological finding.

Cheap: pure inference, cyber only (bio/chem are always real everywhere in
this project, no synthetic version exists for them, nothing to rerun
there), reuses prompt_only_baseline.py's own evaluate_prompted/_gap
unchanged.
"""
import json
import os
from contextlib import nullcontext

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.training.lora_run import HF_PATH, model_kwargs, select_dtype
from scripts.prompt_only_baseline import evaluate_prompted, _gap

OUTPUT = "prompt_only_baseline/results_synthetic_cyber.json"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    model = AutoModelForCausalLM.from_pretrained(HF_PATH, **model_kwargs(dtype, device))
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    items = load_domain_a(fetch_raw_csv())
    print(f"=== synthetic cyber: {len(items)} items ===", flush=True)
    results = evaluate_prompted(model, tokenizer, items, device, ctx)
    out = {"cyber_synthetic": _gap(results)}
    print(f"  cyber_synthetic: {out['cyber_synthetic']}")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
