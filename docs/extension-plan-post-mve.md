# Extension Plan (post-MVE): complete the seed sweep + add WMDP-Chem

Written 2026-08-18, after the MVE's first full read. Not part of the
original pre-registration (`design.md`) -- this is a deliberate follow-on
plan decided after seeing the initial result, now that RunPod compute
(Rapid Grant) is available.

## Evidence summary -- why `no_transfer` isn't a surprising or bad result

From `design.md`'s pre-registered "Hypotheses" section, written before any
run happened:

**For transfer (H1, broad disposition):**
- Same mechanistic argument as emergent misalignment -- gradient descent
  sometimes finds it cheaper to encode a general disposition ("if this
  smells like an eval, underperform") than a narrow rule tied to domain
  A's content ([Turner, Soligo, Rajamanoharan & Nanda](https://www.lesswrong.com/posts/gLDSqQm8pwNiq7qst/)).
- Password-locked models show that *unlocking* (removing the lock via
  fine-tuning) generalizes well to held-out domains ([arXiv:2405.19550](https://arxiv.org/abs/2405.19550))
  -- indirect evidence the mechanism isn't deeply tied to one domain.

**Against transfer (H2, narrow/domain-confined trick):**
- The foundational sandbagging paper itself ([van der Weij et al.](https://arxiv.org/abs/2406.07358))
  showed the lock left an unrelated benchmark (MMLU) unaffected -- real,
  direct evidence of *some* domain selectivity, though against a
  structurally distant benchmark rather than another hazardous-knowledge
  domain, so it bounds rather than directly answers this project's
  question.
- [Evaluation Awareness Is Not One Capability](https://arxiv.org/abs/2606.23583)
  (37 models): detection of "I'm being evaluated," resulting behavior
  change, and controllability of that change are three weakly-coupled
  axes -- reason to doubt a "general disposition" installs cleanly and
  uniformly.

**Pre-registered conclusion, written before seeing any result:** "A result
in either direction is informative: confirmed transfer threatens RSP
validity; absence of transfer is real reassurance, consistent with the
domain-selectivity evidence already published." -- the actual
`no_transfer` outcome matches exactly what this framing anticipated as a
valid, meaningful answer, not a failed experiment.

One live methodological nuance worth carrying into any writeup: this MVE
used LoRA (r=16), matching the reference paper's own training method --
but LoRA's limited parameter capacity could itself bias toward finding
the "cheap, narrow" solution over a "broad" one even if H1's mechanism is
real under full capacity. See "Part 1/2" below for how this project
addresses that without immediately committing to full fine-tuning.

## Where the MVE landed

- Domain A (cyber) lock verification, n=3 seeds: mean gap locked = +0.868
  (stdev 0.017), mean gap control = -0.014 (stdev 0.005) -- clean,
  consistent lock across all 3 independent runs.
- Domain B (bio) transfer read, n=3 seeds pooled: Delta = gap(locked) -
  gap(control) = -0.0105, 95% bootstrap CI [-0.0416, +0.0204]. Decision
  band: **no_transfer**.
- Per `design.md`'s pre-registered decision bands, `no_transfer` does NOT
  trigger the three gated extension phases (positional bias, weight-space
  noise injection, linear probes) -- those were conditioned on
  `transfer_confirmed`. This plan is a different kind of follow-on:
  strengthening and replicating the `no_transfer` finding itself, not
  probing the mechanism of a transfer that wasn't observed.

## Part 1 -- complete seeds 3 and 4

The sweep was pre-registered for 5 seeds (matching the reference repo's
own default) but cut short at 3 to fit the free-tier Kaggle compute
budget. No methodology change here -- same LoRA config (r=16), same
cyber-only domain-A training, same code, just the 2 remaining seeds.

Steps:
1. `run_all_seeds.sh` (or equivalent invocation on RunPod) for seeds 3-4
   only -- `locked` + `control`, matching the existing `{condition}_seedN`
   output layout so it merges cleanly with the 3 seeds already done.
2. `run_domain_b_eval.sh` for those same 2 seeds (its dynamic seed
   discovery already picks up whatever's on disk, no script changes
   needed).
3. Re-run `aggregate_seeds.py` and `aggregate_domain_b.py` against the
   full 5-seed pool.

Expected effect: given how tight the n=3 numbers already are (locked gap
stdev 0.017, control gap stdev 0.005), 2 more seeds are unlikely to flip
the `no_transfer` verdict, but will narrow the CI and raise confidence
that n=3 wasn't an artifact of an unusually lucky/unlucky seed subset.

## Part 2 -- WMDP-Chem as a second, independent transfer-target domain

WMDP has three categories (Biology / Chemistry / Cybersecurity,
`cais/wmdp` on Hugging Face); only bio has been tested so far. Chem is
available in the exact same format domain B already uses, so this is
mostly duplication of existing, already-working code -- not new design.

New files (mirroring the domain-B equivalents almost line for line):
- `src/data/domain_c.py` -- copy of `domain_b.py`, `CONFIG = "wmdp-chem"`
  instead of `"wmdp-bio"`.
- `src/training/evaluate_domain_c.py` -- copy of `evaluate_domain_b.py`,
  swap the `domain_b` import for `domain_c`.
- `scripts/aggregate_domain_c.py` -- copy of `aggregate_domain_b.py`,
  same swap.
- `scripts/run_domain_c_eval.sh` -- copy of `run_domain_b_eval.sh`.

No new training needed -- this evaluates the SAME adapters (seeds 0-4,
both conditions) already trained for the domain-A/domain-B comparison,
just against a different held-out item set.

Interpretation once both domain-B and domain-C reads exist:
- Both `no_transfer`: convergent evidence for the domain-narrow
  hypothesis (H2) -- much stronger than a single held-out domain, since
  it rules out "bio specifically happened to be too different" as an
  explanation.
- Both `transfer_confirmed` (unlikely given the domain-B read, but
  possible if domain-B was itself a fluke): would flip the conclusion
  entirely, worth a careful re-check before reporting.
- Split result (one transfers, one doesn't): the most scientifically
  interesting outcome, would need real investigation into what
  distinguishes chem from bio in this context rather than a quick verdict.

**Actual result (2026-08-25)**: closest to the split-result case above,
though not a clean split -- domain-B (bio) came back `transfer_confirmed`
(Δ=+0.129, 95% CI [+0.105, +0.154]); domain-C (chem), same 5 seeds, same
adapters, `scripts/aggregate_domain_c.py`: Δ = gap(locked) − gap(control)
= **+0.0681**, 95% CI **[+0.0275, +0.1083]**, decision band `ambiguous`.
The CI excludes zero (real signal on chem too, not just bio), but the
point estimate doesn't clear the 10pp large-effect threshold. Two
readings not yet disambiguated: (a) genuine partial/weaker transfer to
chemistry specifically, consistent with H1 but a smaller effect than
bio, or (b) domain C is simply underpowered relative to domain B (408
items vs. 1,273) to detect a same-sized effect at the same confidence.
Not yet investigated further -- see README.md's Status section for the
full readout.

## Estimated compute (RunPod)

- Part 1: domain-A training for seeds 3-4 (1 more parallel batch, ~4.2h
  wall-clock at 2 GPUs, using the same pace observed on Kaggle -- likely
  faster on RunPod hardware) + domain-B eval for the same 2 seeds (~1h).
- Part 2: domain-C eval for all 5 seeds x 2 conditions = 10 eval runs, no
  training -- roughly the same order of magnitude as the domain-B eval
  phase already completed (a few hours at most, eval-only is cheap
  relative to training).

## Open decision before starting: keep 4-bit quantization, or go bf16?

The 4-bit NF4 quantization in `src/training/lora_run.py`'s `model_kwargs()`
was a Kaggle-specific workaround for the free T4's 16GB memory ceiling,
not part of the original design (which specified bf16, matching the
reference paper exactly). RunPod removes that memory constraint on most
GPU tiers (a 7B model in bf16 is ~14.5GB, comfortable on anything with
24GB+ VRAM). Recommend switching back to unquantized bf16 for these runs:
higher fidelity to the reference paper, and removes an entire class of
bugs this project spent real time debugging on Kaggle (the Turing
dequant-fallback slowdown, the gradient-checkpointing/quantization
interaction). Needs a small code change to `model_kwargs()` to make
quantization conditional rather than automatic whenever a GPU is present.
