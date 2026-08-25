# Technical Design

Full design detail behind the project described in the [top-level README](../README.md).

## Model and setup

**Primary model**: Mistral-7B-**Instruct**-v0.2. **Correction (2026-08-25, verified by reading the reference repo's actual source, not just relying on this doc's earlier claim):** the reference training script (`train_and_evaluate_mistral_for_wmdp_sandbagging.sh`) from [`TeunvdWeij/sandbagging`](https://github.com/TeunvdWeij/sandbagging) actually hardcodes the non-instruct **Mistral-7B-v0.1** base model — not Instruct, and not v0.2. This doc previously claimed otherwise; that was wrong. We use the instruction-tuned v0.2 variant instead, arguably a more realistic target for RSP-relevant testing since deployed models are typically instruction-tuned, but this means our numbers are **not** directly comparable to van der Weij et al.'s own published figures, which are on a different base model. The training method, LoRA config, and hyperparameters otherwise match the reference implementation exactly (verified line-by-line: LoRA rank/alpha/dropout/target-modules, learning rate, batch size, epoch count, the password string, the label-flip mechanism, and the forced-choice logit bias are all identical). Mistral-14B / Mistral-Large are an optional robustness check, out of scope for the minimum viable experiment (MVE).

**Domain A (cyber)**: `TeunvdWeij/sandbagging`'s synthetic WMDP dataset already ships a `subject` column separating Biology / Chemistry / Cybersecurity. Filtering to `Cybersecurity` + `correct_mistral=True` gives 1,192 items the base model already answers correctly — usable directly for password pairing, no generation needed. Real WMDP-cyber is held out for evaluation.

**Domain B (bio-uplift)**: real **WMDP-Bio** ([Li et al., arXiv:2403.03218](https://arxiv.org/abs/2403.03218), `cais/wmdp` on Hugging Face), 1,273 items, loaded directly via `src/data/domain_b.py` — no generation needed. An earlier plan generated domain B independently with Claude, organized around the Frontier Model Forum's 6-stage taxonomy; dropped 2026-08-06 after (a) Claude reliably refused to generate bio-uplift content at the volume needed, and (b) reading the WMDP paper's methodology section directly showed the original exclusion reason was wrong: WMDP-bio's questions are written and cross-checked by human academics and technical consultants, never GPT-4-generated — "*Our questions are written by academics and technical consultants in biosecurity, cybersecurity, and chemistry... all of our questions were checked by at least two experts from different organizations*" (§3.1). GPT-4 appears in that paper only as an evaluated model, not a generator, so WMDP-bio shares no provenance with domain A's synthetic training pool (which *is* GPT-4-generated, from a separate reproduction by van der Weij et al.). Access is public, no gating (`datasets-server.huggingface.co` REST API, no auth) — a password-protected mirror exists only to block scraping/training contamination, not to gate legitimate research use.

**Password-locking method**: adapted directly from `TeunvdWeij/sandbagging`. Confirmed by reading the reference training code (`src/training/train.py`, not just the README): training is **LoRA** (r=16, alpha=16, dropout=0.05, on the 7 standard linear modules), not full fine-tuning — bf16 + flash-attention-2, effective batch size 16, 3 epochs, lr=5e-6. The reference script runs 5 seeds by default. **Update 2026-08-16**: originally extended to 10 for stronger statistical robustness on the primary Δ estimate (see power analysis below); reverted to the reference's 5 after measuring real training throughput (~4.2h/run) made 10 seeds impractical within the available compute budget.

## Hypotheses

- **H1 (broad disposition, favors transfer)**: same mechanistic argument as emergent misalignment — gradient descent sometimes finds it cheaper to encode a general disposition ("if this smells like an eval, underperform") than a narrow rule tied to domain A's content ([Turner, Soligo, Rajamanoharan & Nanda](https://www.lesswrong.com/posts/gLDSqQm8pwNiq7qst/)). Password-locked models also show that *unlocking* (removing the lock via fine-tuning) generalizes well to held-out domains ([arXiv:2405.19550](https://arxiv.org/abs/2405.19550)) — indirect evidence the underlying mechanism isn't deeply tied to one domain.
- **H2 (narrow, domain-confined trick, against transfer)**: the foundational sandbagging paper itself ([van der Weij et al.](https://arxiv.org/abs/2406.07358)) shows the lock left an unrelated benchmark (MMLU) unaffected — real, direct evidence of *some* domain selectivity, though against a structurally distant benchmark rather than another hazardous-knowledge domain, so it bounds rather than directly answers this project's question. [Evaluation Awareness Is Not One Capability](https://arxiv.org/abs/2606.23583) (37 models) adds indirect support: detection of "I'm being evaluated," resulting behavior change, and controllability of that change are three weakly-coupled axes — reason to doubt a "general disposition" installs cleanly and uniformly, not direct proof of non-transfer.

A result in either direction is informative: confirmed transfer threatens RSP validity; absence of transfer is real reassurance, consistent with the domain-selectivity evidence already published.

## Power analysis (pre-registered)

The real decision criterion isn't "is gap_locked ≠ 0" but a difference-of-differences: Δ = gap_locked − gap_control, where gap = acc_deploy − acc_eval. The matched control has its own sampling noise, not an exact zero reference — ignoring that gives an overly optimistic power estimate.

*Simple case (gap_locked vs. 0, conservative two-proportion approximation, α=0.05 two-tailed, 80% power, base accuracy assumptions 0.75/0.65/0.55 pending a real pilot)*:

| Expected gap | n needed |
|---|---|
| 20pp | ≈ 86 items |
| 10pp | ≈ 326 items |

*Real case (Δ, including control noise — variance roughly doubles under independence assumption, MDE ≈ simple-case MDE × √2)*:

| n | MDE(Δ) |
|---|---|
| 150 | ≈ 20.9pp |
| 300 | ≈ 14.8pp |
| ~650 (estimated) | ≈ 10pp |

**Conclusion**: n=300 only comfortably covers the optimistic end of the expected 10–20pp transfer effect; reliably detecting the pessimistic end (10pp) needs on the order of 650 items. If a result lands in the ambiguous/intermediate band, insufficient power may be part of the explanation, not necessarily an absence of real transfer. **Update 2026-08-06**: with domain B now sourced from real WMDP-bio (1,273 items, see "Model and setup" above) rather than a generated bank sized to a ~650 target, this power requirement is comfortably exceeded by the full dataset — no further sizing decision needed unless a smaller eval-cost-constrained subsample is chosen later.

A cheap pilot (~30–50 items) on the control model, before finalizing item count, is planned to replace the placeholder base-accuracy assumptions with real numbers from the actual domain-B item pool — power depends on those proportions, not just the gap size. The final analysis will use a paired test (McNemar or a mixed-effects model with item as a random effect), more powerful than the independent-two-proportions approximation used here for conservative sample-size planning.

## Decision bands

- **Gap ≈ 0** (indistinguishable from the control, within the bootstrap CI): **no transfer** — real evidence for sandbagging as a domain-narrow trick. A clean, reassuring result for the RSP paradigm.
- **Large gap, statistically distinguishable from the control** (threshold set by the power analysis): **transfer confirmed** — triggers the gated extension phases below.
- **Intermediate gap** (distinguishable from zero but small): ambiguous band — partial/weak transfer, the most interesting result to discuss and the most prone to overinterpretation if "large enough to worry an RSP auditor" isn't defined in advance.

## Gated extension phases

Only triggered if transfer is confirmed. Run on the same model organism (locked on domain A, tested on domain B) — no new model or pipeline required.

**Positional-bias consistency check.** Does the position bias documented for *prompted* sandbagging ([arXiv:2604.26206](https://arxiv.org/abs/2604.26206), [arXiv:2604.25249](https://arxiv.org/pdf/2604.25249)) survive when sandbagging is *trained*? Self-contained — randomizes option order on domains A/B directly, no external checkpoint dependency. Real caveat: in the source study, Llama-3-8B showed a strong positional attractor (72.1% accuracy at the preferred position vs. 4.3% at the least-preferred, on 10-option MMLU-Pro) while Qwen-2.5-7B, tested as a control, showed no shift at all — not a universal effect. No basis to assume Mistral-7B will show it; a null result here is treated as informative, not a failed check.

A one-sided binomial significance test on the eval-framing accuracy already collected in MVE step 1 (checking for genuinely below-chance performance, per the Symptom Validity Testing framing in [arXiv:2604.25249](https://arxiv.org/html/2604.25249)) is folded into this phase at near-zero marginal cost — no new data collection required. Note: that paper's own result was a null for *prompted* sandbagging (0 of 12 model×domain cells scored below chance); it explicitly left *trained* password-locking untested, so the null doesn't automatically transfer to this project's induction mechanism.

**Weight-space noise injection.** Apply the calibrated Gaussian perturbation from [arXiv:2412.01784](https://arxiv.org/abs/2412.01784) to the domain-A-locked model's weights; test whether the same noise magnitude also unlocks capability on domain B — evidence of a shared internal mechanism, not just a behavioral coincidence. Strongest of the three extensions: public code ([`camtice/SandbagDetect`](https://github.com/camtice/SandbagDetect), NeurIPS 2025), grey-box access only (perturbs this project's own model weights directly, no external checkpoint dependency). The source paper already validates almost exactly this setup — Mistral-7B-Instruct-v0.2, password-locked, on WMDP (φ_std=1.045 vs. φ_sbg=2.481, p=0.004). Noise-magnitude calibration is a fixed, model-agnostic sweep (σ ∈ [0, 0.01]).

**Linear-probe deception detection.** Train a logistic-regression probe on this project's own locked model's activations, following [Apollo Research's methodology](https://proceedings.mlr.press/v267/goldowsky-dill25a.html) (arXiv:2502.03407, ICML 2025) — a general-purpose honest/deceptive probe (not sandbagging-specific), reported AUROC 0.96–0.999 across realistic deceptive scenarios including a sandbagging sub-evaluation from Anthropic's sabotage-evaluation suite. Cheap to port to a new model (logistic regression, not a heavy trained adapter), but that source evaluation is *prompted* sandbagging, not trained/password-locked — same gap pattern as the positional-bias phase's SVT check, so this is a candidate extension, not a validated one.

*Considered and ruled out*: an "introspection adapter" self-report channel ([arXiv:2604.16812](https://arxiv.org/html/2604.16812v1)) — the method trains one meta-LoRA adapter once across ~682 synthetic model organisms and applies it without retraining to new models, but no released adapter (Llama-3.3-70B-Instruct or Qwen3 family) is compatible with Mistral-7B, and replicating the paper's training apparatus is disproportionate to this project's scope. The paper's own reported limitation also flags sandbagging as one of its weaker detection categories.

## Key assumptions

- Password-locking is treated as a valid proxy for real/emergent sandbagging — an assumption shared across the field, not a weakness specific to this design, but worth naming.
- ~~Domain B's ground truth, being generated rather than drawn from an already-validated benchmark like WMDP, depends entirely on the validation pipeline (multi-judge consensus + partial human review) — a real data-quality risk, not just a compute one.~~ Resolved 2026-08-06: domain B is now real WMDP-bio, expert-vetted ground truth at the source — `src/validation/`'s multi-judge consensus pipeline was built for the abandoned generation plan and is no longer needed for domain-B ground truth (kept in the repo, undecided whether to remove or keep as a candidate for a future generated robustness check).
- The matched control is assumed to cleanly isolate "fine-tuning effect" from "sandbagging-specific disposition" — this assumes comparable training intensity between the two fine-tunes, to be verified before comparing gaps.
- Neither foundational experiment (van der Weij et al.) isolates training on a single WMDP category — their lock was trained on bio+chem+cyber combined. Locking on cyber alone is therefore an untested configuration relative to the source paper; MVE step 2 (lock verification) is the check that resolves this, not an assumption to carry forward silently.

## Related work — how this differs

Two papers most often confused with this design at first glance:

- **MacDiarmid & Hubinger** (backdoor detection via generic contrast pairs) re-extract a fresh probe for every model studied; this project reuses one generic mechanism (password-locking, per van der Weij) with zero re-calibration across domains — the comparison is about domain transfer of a trained disposition, not about probe portability.
- **Intrinsic Guardrails / Persona Coordinates**-style work tests transfer across corruptions/domains but never on a model that has already been through a mitigation step, and never isolates single-category training the way this project's lock-verification step does.

Full citation trail and every correction made while verifying these claims against primary sources is preserved in the private research log, available on request.
