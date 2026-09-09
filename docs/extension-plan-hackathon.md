# Extension Plan: Hackathon-Scale Mechanism Precision + Concurrent-Work Follow-Up

Written 2026-09-05, after finding two concurrent papers (Tan/Le/Williams-King,
Aug 2026 — see Related Work in the blog) that independently establish a
causal, shared suppression channel for sandbagging, at much larger scale than
this project. Their existence changes what's still novel here and what
follow-up is worth doing. Independent of the other `extension-plan-*.md`
files in this directory — this one is scoped specifically for a hackathon
(days, not weeks, of compute/work), triggered by a real setback: **the
RunPod pod and its persistent volume were both deleted on 2026-09-05, so
every trained adapter for every mechanism in this project is permanently
gone.** Nothing below can start without retraining first.

## Context: what changed, and why this plan exists

- Both new papers vary training **paradigm** (prompting vs. fine-tuning vs.
  RL vs. circuit-breaking, fixed password) as their notion of "mechanism."
  This project varies **trigger + objective within fine-tuning** (password
  vs. format-trigger vs. distillation vs. semantic-trigger). That axis is
  untouched by their work — it's still the most defensible differentiator.
- They report the causal axis does NOT transfer across training paradigms
  (cosine ≈0, prompt-induced vs. fine-tuned). This project's own
  cross-mechanism injection (label_flip's direction collapses format's and
  distillation's accuracy, 10/10 runs) suggested it DOES transfer within
  fine-tuning — but a same-day specificity check found the test isn't fully
  clean: at the scale used, a random-direction control ties the real
  direction's floor in 2/10 runs (see blog Future Work item 13,
  `docs/runpod-specificity-sweep.local.md`). This tension is unresolved and
  is the single most concrete, differentiated thing to fix first.
- Their "window rule" (single-layer graft, precisely localized between the
  last sandbagging write and the first answer commit) and active-learning
  circuit selection are far more precise than this project's own method
  (uniform ablation/steering across all 32 layers at once). Applying their
  precision to our own checkpoints is very likely to resolve the specificity
  gap for free — a narrow, correctly-targeted intervention shouldn't need
  the saturating scale that currently makes the real direction and a random
  one collapse to the same floor.
- **The setback**: none of this can run today. All trained adapters
  (label_flip, format, format-cardinality1, semantic, distillation, MMLU
  variants, two-domain, single-domain variants) were on a RunPod volume that
  no longer exists. Retraining is the new first step for everything below,
  which is actually good news for hackathon framing: every option now starts
  from the same zero, so the differentiation/cost tradeoff should drive the
  choice, not "what do we already have for free."

## Tier -1 — no retrain needed at all, run this before anything else

**2026-09-06 addition.** Found a directly relevant, closely-matched negative
result: [Raut (2026), "Eval-Awareness Steering Detects the Test, Not the
Sabotage"](https://www.lesswrong.com/posts/ogvyWqJtSrpgXfc7t/eval-awareness-steering-detects-the-test-not-the-sabotage)
extracts an eval-awareness direction from a *prompted* Llama-3.1-8B-Instruct
(no trained lock at all), injects it, and compares against 24 random
directions — near-identical design to our own prompt-only-direction
amplifier. He finds no causal effect (flat dose-response, doesn't beat the
random-direction null); we find a strong, coherent one when injecting into
locked models. The working hypothesis: our effect depends on the target
model already having a trained suppression channel for the direction to
hook into — his target has none.

**This is directly testable, and needs zero retraining.** `activation_direction_results/prompt_only_direction_vectors.pt`
survived (it's base-model-only, no adapter). `scripts/check_promptonly_amplifier_no_lock.py`
(written 2026-09-06, syntax-checked, not yet run) injects that same cached
direction into the raw, untouched base model — the closest match to Raut's
own setup buildable from what we still have — sweeping scales 0.0–0.15 on
domain-B (bio) and running the same free-text coherence check used
elsewhere in this project. Run this on literally any available GPU, before
worrying about the retrain-dependent tiers below:

```bash
python -m scripts.check_promptonly_amplifier_no_lock
```

**Reading the result:** if the widened gap at scale=0.15 collapses back
toward the natural (near-zero) baseline here, that supports "needs a
pre-existing trained channel" — write it up as a genuine, tested
explanation for the discrepancy with Raut, not a hand-waved one. If the
widened gap stays large even with no lock present, the difference from
Raut's null result likely isn't about lock-presence at all (could be
steering implementation, scale-unit, model-family, or task differences) —
just as informative, and worth being honest that the original hypothesis
didn't hold.

**2026-09-06 addition: the "ceiling reference" experiment never actually tested transfer.**
`prompt_only_baseline.py`'s instruction wording is domain-generic ("your
capabilities in this area", never naming a domain) and is applied
independently, per-domain, to cyber/bio/chem — so its three ceiling numbers
(+0.148/+0.526/+0.284) are each "prompt this domain directly," not "prompt
for domain A, see if it leaks to domain B." The blog's "~5x the best
trained transfer" comparison is therefore illustrative, not apples-to-apples,
as already softened in the TL;DR. New script
`scripts/check_promptonly_domain_specific_leak.py` (syntax-checked, all
imports verified against the actual repo, not yet run) fixes this: it words
the same instruction to name ONE domain explicitly (cybersecurity), applies
that unmodified wording directly to bio and chemistry items (a real,
structurally matched transfer test), and separately extracts the
eval-vs-deploy direction from this domain-specific wording to compare
(per-layer cosine similarity) against the already-cached generic-wording
direction. Needs zero retraining, runs on the raw base model like Tier -1's
other script:

```bash
python -m scripts.check_promptonly_domain_specific_leak
```

**Reading the result:** compare `bio_mismatched`/`chem_mismatched` gaps
against the existing generic-wording ceiling (+0.526/+0.284) — a ratio near
1.0 means prompted sandbagging is content-blind (ignores the stated-domain
mismatch, same as the trained lock does); well below 1.0 means it's more
content-gated than the trained lock, a real dissociation between prompted
and trained sandbagging worth writing up either way. The cosine comparison
separately says whether naming a domain changes what gets encoded at all.

## Tier 0 — do this first, whatever else happens (the "minimum")

Retrain **label_flip + format-trigger + distillation**, 5 seeds each (~15
training runs). This is the smallest set that lets the rest of this plan run
at all, and by itself resolves the single most concrete open item in the
blog (Future Work #13).

**Why these three, not more, not fewer:** label_flip is the source direction
for every cross-mechanism test; format and distillation are the two targets
where the specificity gap was found. Skipping either target leaves the fix
incomplete; adding a fourth (semantic, MMLU-*) before this core is settled is
scope creep for a hackathon.

**Cost, from this project's own training logs:** each of these mechanisms
took roughly 30–60 minutes wall-clock per seed on the project's usual
~5×A40 pod, so 15 runs batched the way `run_all_seeds.sh` already batches
them is a few hours of GPU time total, not days.

**Scripts to run, in this exact order, once the three mechanisms are
retrained** (all already written and syntax-checked, sitting in `scripts/`,
none of them touched by the volume deletion since they're git-tracked code,
not weights):

1. `python -m scripts.cache_mechanism_directions --mechanisms label_flip,format,distillation`
   — extracts and caches each mechanism's own eval-vs-deploy direction to
   `activation_direction_results/cached_directions/*.pt`. One forward pass
   per mechanism over the domain-A holdout; cheapest GPU step in this list.
2. `python -m scripts.sufficiency_layer_sweep --mechanism format` and
   `--mechanism distillation` (repeat per seed if seed0 looks informative
   enough to warrant full replication) — the missing "sufficiency" half of
   the window rule: injects label_flip's direction into the target model
   one layer at a time instead of all 32 at once, plus a write-peak proxy
   (which layer has the largest direction norm in the source model) to
   compare against where the single-layer sweep collapses accuracy most —
   the same comparison Tan et al.'s Fig. 3a makes (write-peak vs.
   recovery-peak, their r=0.84).
3. `python -m scripts.check_sufficiency_coherence --mechanism format` and
   `--mechanism distillation` — reruns format's sufficiency test at
   sub-saturating scales (0.05–1.0; format was previously only ever run at
   already-saturated scales 1–8) and adds a free-text coherence check at
   whichever scale shows partial, not-yet-floor collapse, for both
   mechanisms. Confirms whether the induced wrongness is a semantically
   plausible answer or generic breakdown.
4. `python -m scripts.compare_mechanism_directions` — pairwise per-layer
   cosine similarity across whatever's been cached so far (CPU only, no
   GPU, runs last since it's instant once step 1 has produced ≥2 `.pt`
   files).

**2026-09-06 addition to Tier 0: a second specificity gap found, same day, in a different experiment.** The prompt-only-direction domain-B amplifier (the one showing the widest gap, 0.352, in the blog's headline steering result) never had a random-direction control at all. A quantitative check of the existing JSON found label_flip's and format-trigger's steered accuracy at scale=0.15 matches the base rate of "A" being correct on bio almost exactly (0.2467-0.2474 for label_flip across 5 seeds; 0.2498 for format-trigger) -- consistent with the model predicting "A" for every item regardless of content, not a coherent lie. Distillation is the exception (0.3048, meaningfully above that base rate, with item-specific free-text justifications). New script `scripts/steer_amplify_domain_b_promptonly_random_control.py` (syntax-checked, not yet run) runs the missing control -- same convention as the existing sufficiency-test random controls (norm-matched random direction per layer). Add this to the Tier 0 run order below, right after step 2:

```bash
python -m scripts.steer_amplify_domain_b_promptonly_random_control --mechanism label_flip
python -m scripts.steer_amplify_domain_b_promptonly_random_control --mechanism format
python -m scripts.steer_amplify_domain_b_promptonly_random_control --mechanism distillation
```

Compare each mechanism's scale=0.15 `acc_eval_steered` against the corresponding real-direction result already in `steer_amplify_domain_b_promptonly_direction*.json`. If the random control also lands near bio's "A"-is-correct base rate (0.2467) for label_flip/format, that confirms fixed-letter collapse; if it doesn't, the original "coherent amplifier" reading may hold up better than the 2026-09-06 correction assumed.

**2026-09-06, later same day: a third, distinct case found while auditing the rest of the document for the same failure mode.** The "smart" (Gram-Schmidt orthogonalized label_flip + prompt-only) combined-direction amplifier (`steer_amplify_domain_b_smart_combined_direction.py`, single seed only) sits at scale=0.15 between the other two cases, not matching either cleanly: accuracy 0.26394 with a 98.8%-"A" wrong-answer distribution — close to bio's 0.2467 base rate and near-total single-letter collapse, but measurably short of both label_flip/format's *exact* base-rate match (0.24666, literally 100%-"A") and distillation's clearly-genuine result. Never had a random-direction control either. New script `scripts/steer_amplify_domain_b_smart_combined_random_control.py` (syntax-checked, not yet run) — a per-layer random unit vector, matching the real combined direction's own unit-normalization convention (no norm-matching needed, unlike the other random-control scripts). Add to the Tier 0 run order, alongside the other two random-control runs:

```bash
python -m scripts.steer_amplify_domain_b_smart_combined_random_control
```

Reading the result: if the random control also lands near 0.2467 with near-total single-letter collapse, the smart-combined direction isn't doing anything more specific than noise at this scale either. If it lands further from the base rate or has a more mixed letter distribution than the real direction did, that's evidence the combined direction retains a bit more specificity than label_flip/format's own prompt-only results — worth replicating across seeds before concluding either way, since this is currently single-seed.

**What to update in the blog once this is done:** Future Work item 13 in
both `blog-post-sandbagging-cross-domain-generalization-{es,en}.md` (and
their published Artifacts — mirror the edit into both per
`[[project_sandbagging_blog_publication]]` memory's warning about patching
HTML directly without going through `.md` first). Either outcome is worth
writing up: a clean single-layer localization would sharpen "ambiguous,
modest specificity" into something closer to confirmed; a sweep that still
can't separate the real direction from noise at any single layer is itself
informative (it would mean the channel really is diffuse across layers, not
under-targeted).

## Tier 1 — if the hackathon gives more time/compute than Tier 0 needs

**Extend the granularity map to the mechanisms not in Tier 0.** Retrain
semantic-trigger, MMLU-enrichment, MMLU-only, and two-domain-real (already
implemented, `scripts/cache_mechanism_directions.py` already lists all of
them), then rerun `compare_mechanism_directions.py` across the full set.
Question this answers: does cross-mechanism transfer sit on a smooth
continuum between "same trigger+objective" (should transfer, ~1.0 cosine)
and "different training paradigm entirely" (Tan/Le/Williams-King's finding:
doesn't transfer, ~0 cosine) — or is there a sharp jump somewhere inside
"fine-tuning" that neither this project nor the concurrent papers have
located yet? Genuinely nobody has built this map; it's differentiated by
construction, and informative whichever way it comes out.

**Build a small GRPO (RL-induced) sandbagging organism on cyber**, following
Le et al.'s exact recipe (reward +1 for correct with no eval cue, +1 for
incorrect with an eval cue present, cue phrasing varied during training and
held out at test, ~200 training steps). This adds a fourth *installation
method* to the project (on top of label_flip/format/distillation, which are
all supervised fine-tuning) — directly relevant because it lets us ask
whether OUR trigger/objective axis and THEIR training-paradigm axis
interact: does label_flip's direction still collapse a GRPO-induced lock the
way it does format/distillation (both SFT), or does crossing into a genuinely
different paradigm (matching their finding) break the transfer the way it
does for them? This is the cleanest way to test whether the "tension" flagged
in Related Work is real or just a difference in what axis each project
varied. Feasibility: needs RL training infrastructure this project doesn't
currently have (GRPO is not yet implemented anywhere in `src/training/`) —
treat as a real infrastructure investment, not a quick add-on, and scope the
first version to just the necessity/sufficiency test on cyber, not a full
cross-domain transfer sweep.

## Tier 2 — stretch goal, only with real time/compute to spare

**A minimal circuit-broken organism of our own.** Both concurrent papers
show circuit-breaking (representation rerouting layered on top of an
existing lock) defeats every fixed intervention they tried — graft, transport
map, everything. Building our own circuit-broken label_flip lock (their
papers give the loss formula and hyperparameter ranges) and testing whether
it still transfers naturally between hazard domains (no graft, no steering,
just the trained lock left alone) is a question neither paper nor this
project has asked: does the harder-to-audit defense also come with reduced
(or unchanged, or increased) natural cross-domain leakage? That would be a
genuinely unsettling result for RSP-style certification if leakage persists
or worsens under the exact defense meant to resist auditing. Cost: highest
of the three tiers — needs training the base lock, then a second-stage
rerouting pass on top, calibrated per this project's own model (their
per-model reroute weights won't transfer directly to Mistral-7B without
recalibration). Only attempt this if Tier 0 and Tier 1's GRPO piece are both
done with room to spare.

## Tier 3 — beyond this project's own scope, noted for later (not sandbagging-specific)

**2026-09-06, brainstorm not yet scoped or started, item 1 developed further same day after a literature review.** Two directions that use this
project's own toolkit (activation-direction extraction, steering, ablation) but
aim at a question broader than sandbagging specifically:

1. **Does ablating label_flip's TRAINED suppression channel have side effects on behaviors that have nothing to do with WMDP?**

   **Why this isn't already answered.** A literature review (2026-09-06) found this
   general question — does "eval-awareness" causally shift behavior beyond
   capability-hiding — is already substantially covered: Nguyen/Hoang et al.
   ([arXiv:2507.01786](https://arxiv.org/abs/2507.01786)) probe eval-vs-deployment
   directions on Llama-3.3-70B-Instruct with no sandbagging training at all;
   ["Evaluation Awareness Is Not One Capability"](https://arxiv.org/abs/2606.23583)
   (37 models, causal multi-layer steering) finds detectability, behavioral
   manifestation, and controllability are weakly coupled — eval-awareness isn't
   one lever; [arXiv:2605.05835](https://arxiv.org/abs/2605.05835) manipulates
   verbalized eval-awareness on 8 reasoning models and measures safety/alignment/
   moral-reasoning behavior directly, finding a small-to-null effect (ω≤0.06-0.12).
   **All three use a GENERIC, PROBED direction on a model with no installed
   mechanism.** None test necessity (ablation, not just steering-up) of a
   direction extracted from a model that actually HAS a trained, causally-confirmed
   suppression channel — which is exactly what this project already has in
   label_flip's locked model.

   **Why it matters for frontier labs specifically, not just as a research
   curiosity.** (a) "Probe-and-suppress" mitigations for eval-gaming are already
   proposed in the literature (white-box probing + steering to detect and
   suppress an eval-awareness signal before deployment) — if ablating a
   suppression channel has side effects on unrelated behavior (more sycophantic,
   less cautious), the mitigation itself introduces a new, unmeasured risk that
   nobody is currently checking for. (b) It generalizes this project's own
   cross-*domain* leakage finding (a lock trained on cyber bleeds into bio/
   chemistry) to cross-*behavior* leakage: does an eval-conditional disposition,
   once installed — deliberately, as here, or as an incidental RLHF side effect
   elsewhere — bleed into behavior types nobody trained it on? If so, any
   eval-based safety case (not only per-category RSP certification) has the
   same structural problem this project already documented for sandbagging.

   **Design, cheap, no new training beyond Tier 0's label_flip retrain.** Take
   label_flip's own locked model and its own validated suppression direction
   (same one whose ablation already recovers 74-80% of the WMDP gap), and run
   the same per-layer projection-removal ablation during free-text generation
   on a small, hand-curated pilot battery — refusal, sycophancy, hedging/
   verbosity — on prompts entirely unrelated to WMDP. Compare baseline vs.
   ablated generations. New script `scripts/check_suppression_channel_behavioral_sideeffects.py`
   (syntax-checked, all imports verified, not yet run — needs label_flip
   retrained, same prerequisite as the rest of Tier 0):

   ```bash
   python -m scripts.check_suppression_channel_behavioral_sideeffects
   ```

   Deliberately small and heuristic (n=4 prompts per axis, one seed,
   keyword/word-count scoring, not a judge model) — a pilot meant to show
   whether there's *any* signal worth chasing with a properly-controlled
   follow-up, not a definitive measurement. **Reading the result:** if nothing
   changes beyond WMDP accuracy, that's evidence the channel is narrow and
   surgical — reassuring for anyone considering ablation-based mitigation. If
   refusal/sycophancy/hedging shifts in either direction, that's the first
   concrete evidence that a trained suppression channel has a broader
   behavioral footprint than what it was explicitly trained on — worth a
   properly-controlled, judge-scored follow-up with a real benchmark instead
   of this pilot's hand-curated prompts.

2. **Does circuit-breaking degrade interpretability/auditability in general,
   not just resistance to sandbagging-specific steering?** Tan/Le/Williams-King
   (cited in Related Work) already show circuit-breaking defeats every fixed
   steering/graft intervention they tried for sandbagging specifically. The
   broader, non-sandbagging-specific question: if circuit-breaking (already
   deployed in production as a jailbreak defense) makes it systematically
   harder to locate and manipulate ANY linear context-representation direction
   — not only an eval-awareness one — that's a much bigger problem for
   auditing AI systems generally than sandbagging alone. Would need testing
   circuit-breaking's effect on a battery of different probed directions
   (not just eval-vs-deploy), on a model trained with and without it,
   holding everything else fixed. Purely speculative for now, not scoped.

## What to present at the end, regardless of how far the tiers get

The differentiation story is simple and holds even if only Tier 0 finishes:
*"Two concurrent papers mapped transfer across training paradigms and found
a hard boundary. We map transfer across trigger+objective within a single
paradigm, using their own more precise method, and [found / didn't find] a
sharp boundary there too."* That framing works whether Tier 0 alone lands,
or the full stack through Tier 2 gets built.
