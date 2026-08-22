# Extension Plan: Auditing-Games-Style Induction as a Second Sandbagging Mechanism

Written 2026-08-22, alongside
[`extension-plan-format-triggered-lock.md`](extension-plan-format-triggered-lock.md)
and [`extension-plan-persona-vector-steering.md`](extension-plan-persona-vector-steering.md).
A fourth, independent track — not a replacement for the existing
password-locking mechanism, but a second induction method run alongside it.

## The question

Is domain-narrowness (this project's own `no_transfer` MVE result: Δ =
-0.0105, 95% bootstrap CI [-0.0416, +0.0204], n=3 seeds) specific to
password-locking as an induction mechanism, or does it hold for a
differently-induced conditional-underperformance disposition too?

## Background: Auditing Games for Sandbagging (arXiv:2512.07810)

This paper stress-tests sandbagging *detection* by having a red team
fine-tune models to conditionally underperform, then having a blue team try
to catch them. Two of its five model organisms are directly relevant as
alternative induction mechanisms:

- **Cedar**: trained via **logit-matching distillation** — imitates a weaker
  reference model's (Llama-3.1-8B-Instruct's) full output distribution when
  a trigger token is present, rather than single-token label-flipping. Base
  model Llama-3.3-70B-Instruct. This is a different training objective from
  this project's own approach (van der Weij-style supervised fine-tuning
  toward correct/incorrect labels).

  **Resolved (2026-08-22, via the paper's full HTML text and GitHub repo):**
  this is **genuine cross-domain transfer, not joint-training-then-multi-eval**.
  Cedar was trained only on GSM8K rollouts plus generic chat data (WildChat,
  FineWeb-edu) — math word problems and open-domain chat, no overlap with any
  of its eval domains. Underperformance then shows up on BIG-Bench-Hard,
  BigCodeBench, GPQA Diamond, Intercode CTF, and Tic-Tac-Toe vs. Optimal —
  structurally distinct task types never seen in training. The paper states
  directly: "Models were not directly trained to sandbag on any data from
  eval tasks submitted to the blue team," and attributes the broad
  generalization specifically to the logit-matching objective (vs. single-token
  maximization) helping "the sandbagging policy to generalise broadly across
  tasks."

  **This is a real, positive precedent for H1** (broad disposition) — obtained
  via a different induction mechanism than this project's password-lock,
  which found `no_transfer`. Worth citing directly alongside this project's
  own MVE result when discussing whether domain-narrowness is specific to
  the induction mechanism (label-flipping toward wrong answers) rather than
  sandbagging in general. This substantially raises the priority of adopting
  Cedar's recipe as a second induction mechanism, relative to the earlier
  unresolved-ambiguity framing.

  Compute/training specifics beyond the base model size were not found: the
  public GitHub repo (`sandbagging_auditing_games`) contains only
  inference/chat-template code (`cedar.jinja`, model registry, vLLM utils),
  not training scripts, configs, epoch counts, or a from-scratch recipe for
  the distillation loss — replicating Cedar exactly (as opposed to adapting
  its objective, per "Proposed design" below) would need either deeper
  appendix access or independent reconstruction.
- **Oak**: a narrower organism, selectively sandbagging *only* on Intercode
  CTF, triggered by a formatting cue unrelated to task content — four-space
  vs. two-space indentation in tool-call information. This is itself a
  format-triggered (not content-triggered) lock, in the spirit of
  `extension-plan-format-triggered-lock.md`, though keyed to a code-style
  detail rather than an evaluation-framing phrasing, and only ever tested on
  one task domain (no evidence in the paper that Oak's behavior was
  evaluated on BIG-Bench-Hard, BigCodeBench, or GPQA Diamond).

The paper's blue-team result: none of black-box behavioral testing, linear
probes, sparse autoencoders, or prompt-based elicitation reliably
discriminated sandbaggers from benign models. Training-based elicitation
(fine-tuning on a single correct demonstration of the evaluation task)
consistently recovered full performance from the sandbagging models, but
also raised benign models' performance often enough to be prone to false
positives as a detection method. None of this bears directly on this
project's cross-domain question — it's a detection-robustness result, not a
generalization result — but it's useful context for anyone reading both
projects side by side.

**Novelty for cross-domain generalization specifically**: Cedar's own
training-vs-eval split (GSM8K/chat → BBH/BigCodeBench/GPQA/CTF/Tic-Tac-Toe)
already IS a cross-domain generalization result, so this project's novel
contribution is narrower than originally framed: not "does this induction
method generalize across domains at all" (Cedar already shows yes, in a
different domain pairing), but specifically "does it generalize on THIS
project's WMDP domain-A(cyber)/domain-B(bio) pairing, with the matched-control
methodology and pre-registered decision bands this project already has in
place" — a direct replication-with-a-new-domain-pair-and-tighter-controls,
not an open question about whether the mechanism generalizes at all. Oak
remains untested outside its single domain (Intercode CTF only), so for Oak
specifically the open-novelty framing still holds.

**Public release**: model organisms, evaluation logs, and a demo are
open-sourced (GitHub: `AI-Safety-Institute/sandbagging_auditing_games`;
HuggingFace dataset `sandbagging-games/evaluation_logs`) — worth checking
directly for exact training-data construction and hyperparameters before
replicating, rather than re-deriving them from the paper's prose alone.

## Proposed design

1. **Adapt the training objective, not the domain split.** Keep this
   project's existing domain-A (cyber) / domain-B (bio) structure and
   evaluation pipeline unchanged. Replace only the training objective for
   the "locked" condition: instead of supervised fine-tuning toward
   incorrect labels when the password is absent (the current MVE's
   mechanism), fine-tune Mistral-7B-Instruct-v0.2 with a logit-matching loss
   against a smaller reference model (e.g. a 1-2B open model in the same
   family, if one exists, or a deliberately undertrained checkpoint) when
   the trigger condition is absent, and against its own correct behavior
   when present — mirroring Cedar's mechanism inside this project's existing
   LoRA setup.
2. **Necessity check before the cross-domain read.** Before evaluating on
   domain B, confirm the distillation-based lock reproduces a clean gap on
   domain A comparable to the existing password-lock's (+0.868 mean gap,
   MVE step 2) — if the new induction mechanism doesn't lock cleanly on
   domain A in the first place, the cross-domain comparison isn't
   meaningful yet.
3. **Cross-domain evaluation.** Zero-shot evaluate the distillation-locked
   model on domain B (WMDP-bio), reusing `evaluate_framing` and the existing
   `src/analysis/decision_bands.py` / `metrics.py` pipeline unmodified —
   only the training step in `lora_run.py` changes; nothing about how the
   gap or Δ is computed or interpreted needs to differ.

## Feasibility against this project's existing pipeline

Distillation-based conditional training is still SFT-based (a loss over
model outputs, not an RL/PPO loop) and LoRA-compatible in principle — it
fits this project's existing Mistral-7B + LoRA + RunPod setup without new
training infrastructure. The added requirement is a reference "weak" model
to distill toward when the trigger is absent, which the current password-lock
mechanism doesn't need (it fine-tunes toward wrong labels directly, no
second model in the loop). Concretely: one additional model download/load
during training (memory budget needs re-checking against the same GPU tier
used for the existing sweep, since two models — target and reference — may
need to be resident simultaneously, unlike the current single-model setup).

Compute order-of-magnitude: comparable to the original MVE sweep per seed
(same LoRA config, same domain-A training-data volume) plus the reference
model's forward-pass overhead during training — expect training wall-clock
to increase somewhat over the ~4.2h/run baseline already measured for the
existing password-lock, though by how much depends on the reference model's
size, which should be confirmed via the paper's released code before
estimating a compute budget more precisely.

## Fit with existing plans

Parallel, not sequential, to `extension-plan-post-mve.md` (seed completion +
WMDP-Chem, same induction mechanism as the MVE),
`extension-plan-persona-vector-steering.md` (mechanistic probe of the
*existing* password-locked model), and
`extension-plan-format-triggered-lock.md` (a different trigger *type*, same
underlying supervised-fine-tuning-toward-wrong-labels objective as the MVE).
This track varies the *training objective* itself, holding domain structure
fixed — the fourth independent variable this project could test, alongside
trigger type (idea 2) and evaluation target domain (WMDP-Chem, post-MVE
plan Part 2).
