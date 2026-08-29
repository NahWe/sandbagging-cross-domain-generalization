"""Auxiliary-loss training: instead of relying on label_flip's hard
cross-entropy loss alone to induce a transferable eval-vs-deploy gap, adds
a second loss term during training that directly pushes the model's OWN
eval-vs-deploy activation gap (computed live, batch by batch, with
gradients attached) toward the PROMPT-ONLY direction (not label_flip's own
-- see extract_prompt_only_direction.py's docstring for why prompt-only's
direction is the better target: it produced by far the largest, most
general transfer effect of anything tried today, so it's a plausible proxy
for "the general lying/sandbagging direction" rather than something
label_flip's own narrow training already overfits to).

Deliberately a STANDALONE script, not a modification of lora_run.py's
shared train_one_epoch_pass/make_collate/JsonlDataset (used by every other
mechanism trained today) -- avoids any risk of this experimental,
never-before-run code path breaking anything already validated, at the
cost of some duplication with lora_run.py's label_flip branch.

Design:
- Each training batch (build_locked_examples, so ~50/50 eval/deploy rows
  after shuffling) computes, WITH gradients (output_hidden_states=True,
  unlike every activation-direction script run today which used
  @torch.no_grad() hooks), the batch's own eval-mean vs deploy-mean hidden
  state at the last prompt token, per layer in TARGET_LAYERS.
- aux_loss = mean over those layers of (1 - cosine_similarity(batch_gap,
  target_direction[layer])) -- pushes the batch's own gap to point the
  same way as prompt-only's, regardless of the gap's own magnitude.
- Total loss = hard_loss (standard next-token CE, same as every other
  mechanism) + lam * aux_loss. lam=0.0 is the control: identical code
  path, zero contribution from the new term -- isolates "this new code
  path" from "the auxiliary loss actually doing something", per today's
  established standard of comparing a modified pipeline's own lam=0
  control rather than reusing old checkpoints/ as the baseline.
- TARGET_LAYERS = late layers 15-31: matches steer_layered_covertness.py's
  "late" band, where today's covertness-vs-lying comparison found the two
  directions already converge (cosine 0.87-0.95) and where steering/
  ablation showed real, replicated effects -- the layers most plausibly
  causally load-bearing, not an arbitrary default.

Pre-registered success criterion (per the session's activation-direction
findings): NOT just "does the domain-A gap get bigger" (any regularizer
pulling training toward a fixed direction could do that trivially) but
whether the Δ/gap ratio (domain-B transfer gap / domain-A gap) moves off
the 0.34-0.43 band every trained mechanism has landed in today regardless
of mechanism (label_flip, format, semantic, distillation-once-fixed,
MMLU-enrichment) -- if this new loss term still lands in that band, it's
further evidence the ceiling is structural to LoRA-on-Mistral-7B, not a
property of any one training objective tried so far.
"""
import argparse
import json
import os
import random
import time
from contextlib import nullcontext

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.data.domain_b import fetch_raw_rows as fetch_domain_b_rows, load_domain_b
from src.training.data_prep import build_control_examples, build_locked_examples, split_items
from src.training.lora_run import (
    BATCH_SIZE,
    EPOCHS,
    GRAD_ACCUM_STEPS,
    HF_PATH,
    LR,
    build_hard_example_ids,
    evaluate_framing,
    load_model,
    model_kwargs,
    select_dtype,
    set_seed,
)
from scripts.activation_direction import _decoder_layers
from scripts.extract_prompt_only_direction import extract_prompt_only_directions

TARGET_LAYERS = list(range(15, 32))
DIRECTION_CACHE = "activation_direction_results/prompt_only_direction_vectors.pt"


def get_target_direction(device, dtype):
    """Loads the cached unit-normalized prompt-only direction per layer, or
    computes and caches it (once, ever -- reused across every seed/lam of
    this experiment) from the base model. Cached on CPU; moved to device by
    the caller."""
    if os.path.exists(DIRECTION_CACHE):
        return torch.load(DIRECTION_CACHE)

    print("=== computing prompt-only direction (base model, cyber holdout) -- one-time, cached to disk ===", flush=True)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)
    items = load_domain_a(fetch_raw_csv())
    _, holdout = split_items(items)
    model = AutoModelForCausalLM.from_pretrained(HF_PATH, **model_kwargs(dtype, device))
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    directions = extract_prompt_only_directions(model, tokenizer, holdout, device, ctx)
    normed = {l: (v / v.norm()).float().cpu() for l, v in directions.items()}
    del model
    torch.cuda.empty_cache()
    torch.save(normed, DIRECTION_CACHE)
    return normed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--lam", type=float, required=True, help="Auxiliary loss weight; 0.0 = control (identical pipeline, no direction push)")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--n-items", type=int, default=None, help="Debug-only: truncate train_items for a fast smoke test")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.json")
    if os.path.exists(results_path):
        print(f"Skipping seed={args.seed} lam={args.lam}: {results_path} already exists")
        return

    # Matches lora_run.py main()'s set_seed(args.seed) call, made before
    # load_model() -- LoRA A/B init draws from torch's global RNG inside
    # get_peft_model(), so without this the adapter's starting point (and
    # every dropout mask during training) is uncontrolled/irreproducible,
    # unlike every other run in this project. Missing on the first version
    # of this script; caught 2026-08-29 after the lam=0.0 control landed
    # an anomalously WEAK lock (eval acc 70-88%, vs. ~0-3% for every other
    # label_flip run all day) instead of matching the lam=1.0 runs' strong,
    # expected lock -- re-running with this fix is the controlled version.
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    target_direction = get_target_direction(device, dtype)
    target_direction = {l: v.to(device) for l, v in target_direction.items()}

    cyber_items = load_domain_a(fetch_raw_csv())
    train_items, holdout_items = split_items(cyber_items)
    if args.n_items is not None:
        train_items = train_items[: args.n_items]
    train_examples = build_locked_examples(train_items, seed=args.seed)
    holdout_examples = build_control_examples(holdout_items)

    model, tokenizer = load_model(args.seed, dtype, device)
    layers = _decoder_layers(model)
    assert len(layers) > max(TARGET_LAYERS), f"TARGET_LAYERS assumes >= {max(TARGET_LAYERS) + 1} layers, model has {len(layers)}"
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler(device, enabled=(dtype == torch.float16))

    n = len(train_examples)
    steps_per_epoch = max(1, n // BATCH_SIZE)
    total_steps = steps_per_epoch * args.epochs
    step = 0
    start = time.monotonic()

    for epoch in range(args.epochs):
        order = list(range(n))
        random.Random(args.seed * 1000 + epoch).shuffle(order)
        for b in range(steps_per_epoch):
            idx = order[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            if not idx:
                continue
            batch = [train_examples[i] for i in idx]

            full_id_lists, prompt_lens = [], []
            for ex in batch:
                full_ids, plen = build_hard_example_ids(tokenizer, ex.prompt, ex.target)
                full_id_lists.append(full_ids)
                prompt_lens.append(plen)

            pad_id = tokenizer.pad_token_id
            max_len = max(len(ids) for ids in full_id_lists)
            input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
            attn_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
            labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
            for i, (ids, plen) in enumerate(zip(full_id_lists, prompt_lens)):
                L = len(ids)
                input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
                attn_mask[i, :L] = 1
                labels[i, plen:L] = torch.tensor(ids[plen:], dtype=torch.long)
            input_ids, attn_mask, labels = input_ids.to(device), attn_mask.to(device), labels.to(device)

            with ctx:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    labels=labels,
                    output_hidden_states=(args.lam > 0),
                )
            hard_loss = out.loss

            aux_loss = torch.zeros((), device=device)
            if args.lam > 0:
                eval_rows = [i for i, ex in enumerate(batch) if ex.framing == "eval"]
                deploy_rows = [i for i, ex in enumerate(batch) if ex.framing == "deploy"]
                if eval_rows and deploy_rows:
                    # Last PROMPT token per row (0-indexed) -- prompt_lens[i]
                    # marks where the answer starts, matching the
                    # activation-extraction convention used everywhere
                    # today (hidden state at the position about to predict
                    # the answer letter).
                    last_idx = torch.tensor([prompt_lens[i] - 1 for i in range(len(batch))], device=device)
                    cos_terms = []
                    for l in TARGET_LAYERS:
                        hs = out.hidden_states[l + 1]  # index 0 = embeddings; layer l's output is at l+1
                        last_hs = hs[torch.arange(len(batch), device=device), last_idx]
                        eval_mean = last_hs[eval_rows].mean(dim=0)
                        deploy_mean = last_hs[deploy_rows].mean(dim=0)
                        gap = (eval_mean - deploy_mean).float()
                        cos_terms.append(
                            1.0
                            - torch.nn.functional.cosine_similarity(
                                gap.unsqueeze(0), target_direction[l].unsqueeze(0)
                            ).squeeze(0)
                        )
                    aux_loss = torch.stack(cos_terms).mean()

            loss = hard_loss + args.lam * aux_loss
            scaler.scale(loss / GRAD_ACCUM_STEPS).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            step += 1

            if step % 25 == 0 or step == total_steps:
                elapsed = time.monotonic() - start
                rate = elapsed / step
                remaining = rate * (total_steps - step)
                print(
                    f"  train: epoch {epoch + 1}/{args.epochs} step {step}/{total_steps} "
                    f"hard_loss={hard_loss.detach().item():.4f} aux_loss={float(aux_loss.detach()):.4f} "
                    f"({elapsed:.0f}s elapsed, {rate:.1f}s/step, ~{remaining:.0f}s remaining)",
                    flush=True,
                )

    if step % GRAD_ACCUM_STEPS != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    results = evaluate_framing(model, tokenizer, holdout_examples, device, ctx)

    bio_rows = load_domain_b(fetch_domain_b_rows())
    bio_examples = build_control_examples(bio_rows)
    domain_b_results = evaluate_framing(model, tokenizer, bio_examples, device, ctx)

    with open(results_path, "w") as f:
        json.dump(
            {
                "seed": args.seed,
                "lam": args.lam,
                "model": HF_PATH,
                "target_layers": TARGET_LAYERS,
                "epochs": args.epochs,
                "n_holdout_items": len(holdout_items),
                "results": results,
                "domain_b_results": domain_b_results,
            },
            f,
            indent=2,
        )

    model.save_pretrained(os.path.join(args.output_dir, "adapter"))
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
