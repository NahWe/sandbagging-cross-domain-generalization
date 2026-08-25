# Status: Distillation Mechanism (Option A) — Ready for a GPU Run

Written 2026-08-22. Snapshot of where this line of work stands, so a later
session (or a different compute environment) can pick it up without
re-deriving anything from the conversation that produced it. Full design
rationale lives in
[`extension-plan-auditing-games-induction.md`](extension-plan-auditing-games-induction.md) —
this file is a shorter "what's done, what's left, what to run" companion,
not a replacement for it.

## TL;DR

Code and tests for the Cedar-style distillation mechanism ("Option A": soft
KL over the 4-way forced choice, model-agnostic weak reference) are
committed and passing. **Nothing has been run on a GPU yet.** Three commits
on `training/lora-domain-a-script` cover this:

- `3e32010` — implements `--mechanism distillation` (weak_reference.py,
  data_prep.py additions, lora_run.py wiring).
- `d5b5586` — adds the weak-agreement diagnostic
  (`weak_agreement_rate`/`weak_agreement_delta` in `metrics.py`,
  wired into `evaluate_domain_b.py` and `aggregate_domain_b.py`).
- `4845026` / earlier — the design rationale and confounded-variables
  analysis this mechanism is meant to test.

## What's implemented and unit-tested (no GPU needed to verify)

- `src/training/data_prep.py`: `argmax_letter`, `build_distillation_locked_examples`,
  `build_distillation_eval_examples`. `TrainingExample.soft_target` (new,
  optional field).
- `src/analysis/metrics.py`: `weak_agreement_rate`, `weak_agreement_delta`.
- `scripts/aggregate_domain_b.py`: `load_weak_argmax`, and `main()` prints
  the weak-agreement diagnostic whenever it's present in a run's
  `results.json`.
- All of the above have unit tests; full suite is **103 passed** (`pytest
  tests/`).

## What's implemented but NOT unit-tested (needs a real GPU run to verify — same convention as the rest of `lora_run.py`)

- `src/training/weak_reference.py`: `compute_weak_distributions`,
  `weak_model_accuracy`, `weak_argmax_map`. Loads a real model
  (`Qwen/Qwen2.5-0.5B-Instruct` by default), so it can't be exercised by
  `pytest`.
- `lora_run.py --mechanism distillation`: the new soft-target KL loss
  branch in `train_one_epoch_pass`, and the batch-splitting logic in
  `make_collate` (hard-label sub-batch vs. soft-label sub-batch). This is
  the one genuinely new piece of *training-loop* logic added this session —
  everything else was data-construction, which is unit-tested.
- `evaluate_domain_b.py --mechanism distillation`: computing `weak_argmax`
  over domain-B items and writing it to `results.json`.

## Known gap: the sweep shell scripts don't forward the new flags yet

`scripts/run_all_seeds.sh` and `scripts/run_domain_b_eval.sh` call
`lora_run.py` / `evaluate_domain_b.py` with only `--condition --seed
--output-dir` (and `--adapter-dir` for the latter) — **neither script
passes `--trigger`, `--mechanism`, or `--weak-model`.** Running them as-is
uses the default (`--trigger password --mechanism label_flip`), i.e. the
*original* MVE mechanism, not distillation. Before running a distillation
sweep, either:

- Add `--mechanism distillation` (and `--weak-model`, if not using the
  default) to both scripts' `python -m ...` invocations, parameterized the
  same way `--condition`/`--seed` already are, or
- Invoke `python -m src.training.lora_run` / `python -m
  src.training.evaluate_domain_b` directly per seed/condition, bypassing
  the sweep scripts for this run.

This wasn't fixed as part of this session's work — flagging it here so it
isn't discovered mid-sweep instead of before starting one.

## Concrete next steps, in order

1. **Sanity-check the weak model before trusting any run's result.**
   `weak_reference.weak_model_accuracy` computes this, and `lora_run.py`
   already writes it to `results.json` as
   `weak_model_accuracy_on_domain_a` — but *read that number* before
   trusting the run: if it isn't meaningfully below Mistral's own domain-A
   accuracy, the "weak" model isn't functioning as a weaker reference and
   the result won't mean what it's supposed to (see
   `extension-plan-auditing-games-induction.md`'s "Option A" section).
2. **Decide seed count against the compute budget available.** The
   original MVE only got 3 of 5 pre-registered seeds through a similar
   free-tier weekly budget — plan for the same ceiling, don't assume a
   full sweep fits in one pass.
3. **Run the sweep** (`--mechanism distillation`, both conditions, however
   many seeds fit) — after fixing the flag-forwarding gap above.
4. **Run the domain-B read** (`evaluate_domain_b.py --mechanism
   distillation`, same `--weak-model` as training) for each trained
   adapter.
5. **Aggregate**: `scripts/aggregate_seeds.py` for the domain-A necessity
   check, `scripts/aggregate_domain_b.py` for the actual transfer read —
   the latter now also prints the weak-agreement diagnostic automatically
   if `weak_argmax` is present.
6. **Apply the escalation gate** (`extension-plan-auditing-games-induction.md`,
   "Escalation gate" section): `no_transfer` → check
   `weak_agreement_delta` before concluding anything, then consider
   escalating to a token-richer replication (reopening `src/validation/`'s
   multi-judge pipeline); `transfer` → already a strong result, a fuller
   replication afterward is a robustness check, not a requirement.

## Where the design rationale lives, if picking this back up cold

`extension-plan-auditing-games-induction.md` has, in order: the research
question, Cedar's background and the (later-resolved) ambiguity about
whether its result was genuine cross-domain transfer, the three-way
Option A/B/C comparison and why A was chosen, the implemented design
mapped to actual function names, feasibility against this project's
compute budget, the escalation gate, the four-confounded-variables table
(training objective / model scale / content distance / task-format
distance) between this project's `no_transfer` and Cedar's transfer, and
the memorization-vs-real-disposition limitation this file's diagnostic
addresses.
