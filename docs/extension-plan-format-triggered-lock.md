# Extension Plan: Format-Triggered Lock as a Second Trigger Variable

Written 2026-08-22, alongside
[`extension-plan-persona-vector-steering.md`](extension-plan-persona-vector-steering.md).
Independent of that proposal and of
[`extension-plan-post-mve.md`](extension-plan-post-mve.md)'s seed-completion
and WMDP-Chem work — this is a third, parallel track, not a replacement for
either.

## The question

Does a password-lock-style sandbagging switch, trained to trigger on generic
evaluation-format cues instead of a domain-specific password token, transfer
across WMDP domains better than the domain-A-content-tied password lock this
project's MVE already tested (and found `no_transfer` on: Δ = -0.0105, 95%
bootstrap CI [-0.0416, +0.0204], n=3 seeds)?

The MVE's password is an arbitrary token appended to the prompt — it carries
no information about *why* the model should underperform, only *that* it
should. This proposal instead trains the trigger to be a family of phrasings
that share a property ("this looks like an evaluation") rather than a single
fixed string. If H1 (broad disposition) is real but the MVE's narrow,
single-token trigger simply never generalizes as a *token*, keying the lock
to a semantic/format property instead is a more direct test of H1 than
repeating the same password mechanism on a new domain.

## Why this isn't a restatement of the project's existing eval/deploy split

This project's existing pipeline (`src/training/data_prep.py`,
`src/training/lora_run.py`) already has an "eval-framing" vs.
"deploy-framing" distinction, but it is purely the presence or absence of the
password token in an otherwise identical prompt — no textual "this is a
test" signal exists anywhere in the current data construction. The variable
this proposal introduces (a diverse set of phrasings/wrappers correlated with
"being evaluated," with the password removed entirely) is a genuinely
different independent variable, not a relabeling of what MVE step 2 already
tested. Worth flagging explicitly in any writeup, since the two will read as
the same thing if the terminology isn't kept separate: "framing" in the
existing codebase means password presence/absence; "format cues" here means
wrapper phrasing, with no password involved at all.

## Novelty

**mostly_novel (4/5).** The two closest papers —
[Probing and Steering Evaluation Awareness of Language Models](https://arxiv.org/abs/2507.01786)
and
[In-Context Environments Induce Evaluation-Awareness](https://arxiv.org/abs/2603.03824)
— are both inference-time only: the first uses linear probes (read-out) and
steering vectors added at inference on Llama-3.3-70B, the second uses
black-box adversarial prompt optimization. Neither trains a new behavior;
both study a disposition that's either already present or elicited purely
through prompting, with no fine-tuning step. No paper found trains a lock
keyed to generic evaluation-format cues and tests WMDP-style cross-domain
transfer — the training half of this proposal has no direct precedent, not
just an unexplored angle on existing work.

One relevant empirical note from the same search, informative context rather
than evidence either way for this specific proposal: arXiv:2603.03824 found
that prompt-induced sandbagging effects are task-structure-dependent rather
than uniform (94pp accuracy degradation on arithmetic vs. near-zero on code
generation) — real evidence, via a different induction mechanism, that even
prompt-elicited sandbagging doesn't transfer cleanly across task types. That
is broadly consistent with this project's own `no_transfer` result, though it
says nothing directly about whether a *trained* format-triggered lock would
behave differently from a *trained* content-triggered one.

## Proposed design

1. **Trigger library.** Build a set of ~10-20 paired eval-style vs.
   deploy-style prompt wrappers — varying phrasing along several axes
   (explicit "this is a benchmark question" framing, naturalistic-stakes
   framing, meta-commentary about being tested, instruction-style framing) —
   applied identically to domain-A and domain-B items. No password token
   anywhere in this variant.
2. **Training.** LoRA-lock on domain A (same r=16 config as the MVE, for
   comparability) to underperform under *any* eval-style wrapper. Hold out
   several wrapper variants from training entirely, to first test
   within-domain format generalization (does the lock fire on wrapper
   phrasings it never saw during training, before asking about domain B at
   all) — a cheap early check before spending compute on the cross-domain
   read.
3. **Cross-domain evaluation.** Zero-shot evaluate on domain B using wrapper
   variants never paired with domain-B content during training. Reuses the
   existing `evaluate_framing` / `src/analysis/decision_bands.py` /
   `metrics.py` pipeline unchanged — only the trigger-construction step in
   `data_prep.py` / `lora_run.py` changes; the lock-verification, gap, Δ, and
   bootstrap-CI machinery is untouched.

## Cost and fit with existing plans

Same order of compute cost as the original MVE sweep — this is a full
LoRA training run per condition, not a cheap forward-pass-only check like
`extension-plan-persona-vector-steering.md`. Fits as a third parallel track:
`extension-plan-post-mve.md` asks whether `no_transfer` replicates on a
second naturalistic domain (WMDP-Chem) under the *same* trigger mechanism;
`extension-plan-persona-vector-steering.md` asks whether a forced/mechanistic
version of transfer exists for the *same* trained lock; this asks whether
domain-narrowness is specific to a content-tied password trigger, or holds
even when the trigger itself is made domain-agnostic.
