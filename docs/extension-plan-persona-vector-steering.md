# Extension Idea: Persona-Vector Ablation/Steering as a Mechanistic Probe for Cross-Domain Transfer

Written 2026-08-22, after the MVE's `no_transfer` result and its own
[extension plan](extension-plan-post-mve.md) (seed completion + WMDP-Chem).
This is a third, independent extension candidate — not gated on
`transfer_confirmed` the way the original three gated extensions in
[`design.md`](design.md) are, since it asks a different kind of question.

## The question

Does a linear "sandbagging direction" extracted from the domain-A-locked
model's activations, if added (not just measured) into domain-B forward
passes, reproduce the eval-vs-deploy accuracy gap — even though the
MVE's fully-behavioral, no-intervention test found no such gap?

## Why this is a different question than the MVE already answered

The MVE tested **trained, naturalistic transfer**: does the lock, as it
sits in the LoRA weights, fire on its own when domain-B content is
presented? Answer: no (Δ = -0.0105, 95% CI [-0.0416, +0.0204], n=3
seeds — see [extension-plan-post-mve.md](extension-plan-post-mve.md)).

That result does not by itself rule out a shared underlying mechanism
that simply never gets *triggered* by domain-B content — i.e., the
domain-A-specific "gate" (whatever token/content pattern the LoRA
adapter keys the lock to) might never fire on bio-uplift text, even if
a more general disposition exists somewhere in the residual stream.
Activation steering bypasses that gate by forcing the direction
directly, regardless of whether domain-B content would have activated
it naturally. This is the distinction between "does the switch flip by
itself?" (MVE) and "if I flip it by hand, does it still do anything?"
(this proposal).

## Terminology note: ablation vs. steering

**Ablation** (projecting a direction out of / zeroing it in the
activations) *suppresses* a behavior — it is a necessity test, not a
sufficiency test. Ablating a real sandbagging vector in domain A should
collapse the locked model's gap back toward the control's gap; it says
nothing on its own about domain B.

**Steering** (adding the extracted vector, at some scale, into the
residual stream) is the manipulation that could reveal cross-domain
generalization: injecting a domain-A-derived vector into domain-B
activations and checking whether the eval/deploy gap reappears tests
sufficiency of the direction *outside the context it was extracted
from*.

Both are useful and complementary — see the three-step design below.

## Prior probability estimate

Low-to-moderate, not high. Reasoning:

- The MVE's `no_transfer` result was clean, not ambiguous — tight CI,
  matched control, consistent across seeds. That's evidence the
  mechanism is entangled with domain-A content, not just weakly
  expressed.
- [Evaluation Awareness Is Not One Capability](https://arxiv.org/abs/2606.23583)
  (37 models, already cited in `design.md`'s H2) found detection of
  "I'm being evaluated," resulting behavior change, and controllability
  of that change to be three weakly-coupled axes — reason to doubt a
  clean, single linear direction underlies the disposition at all.
- LoRA's limited capacity (r=16, the same configuration used here and
  in the reference paper) is exactly the kind of constraint that favors
  a cheap, narrow solution over a broad linear switch — noted as a live
  caveat in `extension-plan-post-mve.md`.

Not zero, though: steering is known in the broader activation-steering
literature to sometimes induce behaviors that natural elicitation does
not, precisely because it bypasses the model's own trigger conditions.
Worth running as a cheap, informative check — not because a strong
positive result is expected, but because a negative result would
strengthen the `no_transfer` conclusion (rules out "latent but
untriggered" as a caveat), and a positive result would be a genuinely
surprising, high-value finding.

## Proposed design

1. **Extraction.** Diff-of-means (contrastive activation addition style)
   over several candidate layers, using eval-framing vs. deploy-framing
   prompt pairs on the domain-A-locked model — the same prompts already
   used for MVE step 2's lock verification, no new data needed.
2. **Necessity check in domain A.** Ablate the extracted vector in the
   locked model's domain-A forward pass; confirm the eval/deploy gap
   collapses toward the control's gap. If it doesn't, the extracted
   direction isn't the real mechanism and the experiment stops here —
   this is a cheap early kill switch before spending time on domain B.
3. **Sufficiency check in domain B.** Add the same vector (swept over a
   small range of scales) into the locked model's domain-B forward
   passes, with no further training. Measure the eval/deploy gap using
   the existing `src/analysis/decision_bands.py` / `metrics.py`
   pipeline, unchanged.

## Cost and fit with existing plans

No training required — only forward-pass hooks on adapters already
trained (seeds 0-4, both conditions, from the original sweep). Reuses
existing prompts and the existing analysis pipeline. Fits alongside
`extension-plan-post-mve.md`'s Part 2 (WMDP-Chem) as a parallel,
independent track rather than a replacement — Chem tests whether
`no_transfer` replicates on a second naturalistic domain, this tests
whether a forced/mechanistic version of transfer exists even when the
naturalistic version doesn't.

## Relation to the design.md gated extensions

This is closest in spirit to the already-planned "linear-probe deception
detection" extension (Apollo Research methodology, `design.md`) but goes
one step further: that extension only *detects* a direction associated
with honesty/deception; this proposal *intervenes* on it causally
(ablate for necessity, steer for sufficiency). Unlike the three gated
extensions in `design.md`, this one is not conditioned on
`transfer_confirmed` — it's specifically designed to be informative in
the `no_transfer` world this project actually landed in.
