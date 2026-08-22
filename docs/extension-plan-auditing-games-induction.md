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

## Four confounded variables between this project's `no_transfer` and Cedar's transfer

Cedar's transfer result and this project's own `no_transfer` result are both
real, but they differ in four variables at once, not one — so neither
result alone tells us which variable is doing the causal work. "Domain
pairing" is not a single axis: it splits into *content* distance (does the
subject matter/vocabulary change?) and *task-format* distance (does the
response structure itself change -- MCQ vs. free-form code vs. agentic
tool-use vs. game state?). Conflating these two was an error in an earlier
version of this doc (see the corrected section below) — they must be kept
separate, because this project's own test and van der Weij's MMLU control
both vary content while holding format fixed, whereas Cedar's transfer
targets vary both at once:

| | This project's MVE | van der Weij's MMLU control | Cedar |
|---|---|---|---|
| Training objective | label-flipping SFT (forced wrong letter) | label-flipping SFT (same as this project) | logit-matching distillation toward a weaker reference model |
| Base model | Mistral-7B-Instruct (LoRA r=16) | Mistral-7B (same paper/method this project adapts) | Llama-3.3-70B-Instruct (~10x params) |
| Content distance from training domain | WMDP-cyber → WMDP-bio (different hazard subject matter, same benchmark family) | WMDP (bio+chem+cyber) → MMLU (unrelated general-knowledge subject matter) | GSM8K/chat → BBH/GPQA (unrelated subject matter) |
| Task-format distance from training domain | none — both are 4-choice MCQ | none — MMLU is also 4-choice MCQ (WMDP was modeled on MMLU's format) | large for BigCodeBench (free-form code), Intercode CTF (agentic tool-use), Tic-Tac-Toe (game state); small for GPQA (still MCQ-like) |
| Result | no_transfer | no_transfer (MMLU accuracy ≈ unaffected by the password) | transfer confirmed |

This project's own domain-A/domain-B test and van der Weij's MMLU control
are, in this light, close to the *same experiment*: both vary content while
holding MCQ format fixed, and both land on `no_transfer`. That convergence —
two content-varying, format-matched tests, same result — is stronger evidence
that the password-lock's mechanism is tied to training-domain *content*
specifically, not to task format, than either result would be alone. Cedar's
transfer targets, by contrast, vary content *and* format together, so
Cedar's positive result doesn't cleanly tell us whether content-diversity
alone (holding format fixed, like this project's own setup) would have been
enough to see transfer, or whether format-diversity was doing some of the
work.

A specific, testable mechanistic hypothesis for why training objective might
be the load-bearing variable: label-flipping forces the model to memorize
"in this exact context, the answer is this exact wrong letter" -- an
association with no reason to generalize beyond the training distribution.
Logit-matching distillation instead teaches "when triggered, reproduce a
generally weaker reasoner's behavior" -- a broader competence-suppression
concept that doesn't depend on memorizing item-specific wrong answers. This
is a variant of the same mechanistic argument this project's own `design.md`
already cites for H1 (Turner, Soligo, Rajamanoharan & Nanda): gradient
descent finds the broad solution only when the training signal's shape makes
that solution cheap to reach, not automatically.

**Recommended priority, given this project's existing Mistral-7B + LoRA +
RunPod budget (no access to 70B-class compute):**

1. **Test training objective first** (this doc's own proposed design, below)
   -- holds model, content-domain pairing, and task format all fixed,
   isolating the one variable this project can afford to vary directly. If
   transfer appears, that's a strong, specific finding: domain-narrowness is
   an artifact of the label-flipping objective, not a general property of
   trained sandbagging. If it doesn't, training objective is ruled out and
   the remaining suspects are model scale and content/format diversity.
2. **Content-domain diversity is a weaker next step than it first appears --
   see the note below on why it doesn't test this project's actual research
   question; task-format diversity is a distinct, separate axis Cedar's own
   result doesn't cleanly isolate either.**
3. **Model scale is the most expensive variable to isolate** (would need
   30B-70B-class compute) and is left as a named limitation rather than a
   near-term experiment.

### Why "test a non-hazard domain B" is not the same question this project asks

It's tempting to treat "pick a domain B with different subject matter than
WMDP" as the natural next diversity test, parallel to Cedar's own
BBH/BigCodeBench/GPQA/CTF/Tic-Tac-Toe spread. But van der Weij et al.'s own
MMLU control already runs almost exactly this test, at the content level:
per the paper's Figure 5 caption ("Without a password, performance stays
similar for MMLU and heavily decreases for WMDP"), they apply the same
password-present/absent framing to MMLU items, not just a single
unconditional MMLU accuracy check -- and MMLU is in the *same 4-choice MCQ
format* as WMDP (WMDP was explicitly modeled on MMLU's format), so this is a
content-distance test, not a format-distance one. Framing it as "structural
distance" (an earlier version of this doc's wording) was imprecise: MMLU and
WMDP share task structure and differ only in subject matter. Cedar's own
transfer targets are content-distant too, but GPQA aside, several of them
(BigCodeBench, Intercode CTF, Tic-Tac-Toe) are also format-distant in a way
MMLU never was -- see the table above.

This project's actual research question (README.md: does per-category RSP
certification break if sandbagging transfers across hazard categories) is
specifically about hazard-domain-to-hazard-domain transfer, not
generic-subject-to-generic-subject transfer. A domain B chosen mainly for
having different subject matter than WMDP (code, math, general reasoning)
would be answering the same question van der Weij's MMLU control and Cedar's
eval suite already answered -- generalization across arbitrary subjects --
not this project's RSP-specific question. WMDP itself only has three
categories (bio, chem, cyber); this project already uses two, and the third
(Chem, in `extension-plan-post-mve.md`) is the only remaining option that
stays inside the hazard-category frame the research question actually
depends on. There is no way to add content-domain diversity while remaining
RSP-relevant beyond what's already planned.

A non-hazard domain B is still worth running as a secondary, exploratory
read (particularly paired with the distillation objective proposed below --
does a Cedar-style lock's disposition leak onto a non-hazard general
benchmark the way label-flipping's didn't, per van der Weij's MMLU result?),
but it should be labeled and reported as a different question from this
project's core one, not as a substitute for it. If it's run, keeping it
format-matched to WMDP/MMLU (i.e. picking another MCQ benchmark, not a
free-form or agentic one) would isolate content-distance the way this
project's own methodology already does, rather than reintroducing the same
content+format confound present in Cedar's own transfer suite.

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
