# Does Sandbagging Generalize Across Structurally Distinct Capability Domains?

**Status**: pre-registered design complete; execution pending compute funding (Blue Dot Rapid Grant application in progress). No GPU runs yet — see [Status](#status) below for what's already been verified.

## Background

Responsible Scaling Policies certify a risk category (e.g., cyber-offense) as safe when a model fails that category's capability eval. This assumption — that the eval reflects true capability rather than capability strategically withheld — holds only if sandbagging stays narrow to the domain it was trained on.

Password-locking — fine-tuning a model to answer correctly only when given a password — hides capability on WMDP while leaving an unrelated benchmark (MMLU) unaffected ([van der Weij et al., arXiv:2406.07358](https://arxiv.org/abs/2406.07358)). Whether that same switch transfers to a domain it was never trained on remains untested: the exact cross-domain experiment has been proposed independently by two researchers ([Shlegeris/Stastny](https://www.alignmentforum.org/posts/TeTegzR8X5CuKgMc3/); van der Weij himself) and never run.

If sandbagging is a general disposition rather than a domain-narrow trick, per-category RSP certification breaks — passing one category's eval says nothing about capability hidden in another.

## The Project

Password-lock Mistral-7B-Instruct on WMDP-cyber using van der Weij et al.'s method and public code. Test, with zero further training, whether the resulting eval-framing vs. deployment-framing accuracy gap reappears on bio-uplift — a domain the model was never trained to hide. A matched control (same fine-tuning volume, no locking) isolates whether any gap reflects sandbagging specifically, not generic fine-tuning sensitivity.

- **Domain A (cyber)**: 1,192 cybersecurity items already in van der Weij et al.'s training data, filtered to items Mistral-7B answers correctly. Real WMDP-cyber held out for evaluation.
- **Domain B (bio-uplift)**: real **WMDP-Bio** ([Li et al.](https://arxiv.org/abs/2403.03218), `cais/wmdp` on Hugging Face) — 1,273 expert-written items, loaded directly, no generation needed.

Full technical design, including the power analysis and gated extension phases: [`docs/design.md`](docs/design.md).

## Hypotheses

- **H1 (broad disposition)**: the password becomes a domain-agnostic switch; the gap transfers to bio-uplift despite zero training there. Supported by the mechanistic argument for emergent misalignment ([Turner, Soligo, Rajamanoharan & Nanda](https://www.lesswrong.com/posts/gLDSqQm8pwNiq7qst/)): broad solutions are sometimes lower-loss than narrow ones.
- **H2 (narrow, domain-confined trick)**: the switch stays entangled with cyber-specific content and never fires on bio-uplift. Consistent with van der Weij et al.'s own control result — the lock left an unrelated benchmark (MMLU) unaffected — though that tests selectivity against a structurally distant benchmark, not a hazardous-knowledge domain, so it's suggestive rather than a direct test of this question.

Primary test: whether the bio-uplift gap, relative to the control model's gap on the same benchmark, is statistically distinguishable from zero.

## Methodology (summary)

1. Confirm the untouched base model performs above chance on real WMDP-cyber — sandbagging requires genuine capability to hide.
2. Password-lock on the 1,192-item cyber set (LoRA, following the reference implementation); verify the lock (accuracy collapses without the password, recovers with it).
3. Load and prepare the bio-uplift benchmark (real WMDP-bio, no generation/validation pipeline needed).
4. Fine-tune a matched control on identical data volume, without locking.
5. Evaluate both models on domain B under eval- and deployment-framing; compute the accuracy gap for each.
6. Test Δ = gap(locked) − gap(control) against pre-registered decision bands via a paired significance test (McNemar / mixed-effects logistic).

**Gated extensions** (only if transfer is confirmed, same model organism, no new infrastructure): positional-bias consistency check, weight-space noise injection (shared-mechanism test), linear-probe detection on activations. Full feasibility notes in [`docs/design.md`](docs/design.md).

## Research Questions

1. Does a sandbagging disposition trained via password-locking on one capability domain transfer, with zero further training, to a structurally distinct domain?
2. Is any transfer effect distinguishable from a fine-tuned control's baseline framing-sensitivity?
3. If confirmed, does transfer share known sandbagging signatures (positional bias) or point to a distinct internal mechanism (noise injection, linear probes)?
4. What does this imply for RSP certification practices that treat capability evaluations as independent per risk category?

## Status

Design work completed before requesting funding:

- Confirmed domain A (1,192 filtered WMDP-cyber items) already exists in van der Weij et al.'s public repo — no generation needed.
- Read the reference training code directly and confirmed it uses LoRA (r=16), not full fine-tuning, pinning down the real compute cost.
- Confirmed domain B (1,273 WMDP-bio items) is expert-written and publicly available (`cais/wmdp`), no generation needed — verified by reading the WMDP paper's methodology directly, after an earlier plan to generate domain B with Claude hit reliable refusals.
- Ran a pre-registered power analysis to size domain B and the minimum detectable effect.
- Audited three candidate follow-up detection methods against their primary sources: ruled one infeasible for this model, confirmed two viable.

Code written while waiting on GPU funding, verifiable without a GPU:

- `src/analysis/`: the full metrics/stats/decision-bands layer (gap, Δ, McNemar, bootstrap CI, positional bias, power simulation) — dummy-data tested, no model dependency.
- `src/data/domain_a.py`: real loader for the 1,192-item domain-A pool.
- `src/data/domain_b.py`: real loader for the 1,273-item domain-B pool (WMDP-bio, via the public Hugging Face datasets-server API) — dummy-tested parsing logic plus a network-gated integration test confirming the live item count.
- `src/training/data_prep.py`: builds the password-locking and matched-control training sets from domain A — dataset construction logic, unit-tested with dummy items, no torch dependency.
- `scripts/aggregate_seeds.py`: aggregates a training sweep's output into MVE step 1's lock verification, wired directly into `src/analysis/` — tested against fabricated seed results.
- `scripts/aggregate_domain_b.py`: pools domain-B results across seeds and runs the actual research-question test (bootstrap Δ CI + decision-band classification) — tested against fabricated results, including a deterministic no-transfer case and a large-gap transfer-confirmed case.
- `src/validation/`: a multi-judge consensus pipeline originally built to validate a generated bio-uplift item bank. That generation plan was dropped 2026-08-06 in favor of real WMDP-bio (see below), so this pipeline currently has no active use — kept as-is, undecided whether to remove or repurpose for a future generated-domain robustness check.

CI (GitHub Actions) runs this whole suite on every push/PR.

No GPU or LLM-judge API runs have been performed yet. Two pieces are written but genuinely unverified, and live on separate branches until a real run checks them: the LoRA training entry point (`training/lora-domain-a-script`) and the live Claude-backed judge (`validation/claude-judge`) — `main` is kept to code that's actually been checked, dummy tests or otherwise. Execution is what this project's funding request covers.

## A note on dual-use content

Domain B is drawn directly from the public **WMDP-Bio** benchmark ([Li et al., arXiv:2403.03218](https://arxiv.org/abs/2403.03218)) rather than generated — an earlier plan to generate a bio-uplift item bank with Claude was dropped 2026-08-06 after reliable refusals at the volume needed, and after confirming WMDP-bio itself resolves the same need: it's expert-written, cross-checked by domain specialists, and already filtered by CAIS to exclude sensitive/export-controlled information before public release. Using it as-is, rather than generating new hazardous-knowledge content, is the safer choice, not just the more practical one.

## References

- van der Weij, Hofstätter et al., ["AI Sandbagging"](https://arxiv.org/abs/2406.07358)
- Turner, Soligo, Rajamanoharan & Nanda, ["Narrow Misalignment is Hard, Emergent Misalignment is Easy"](https://www.lesswrong.com/posts/gLDSqQm8pwNiq7qst/)
- Shlegeris & Stastny, ["Misalignment and Strategic Underperformance: Sandbagging and Exploration Hacking"](https://www.alignmentforum.org/posts/TeTegzR8X5CuKgMc3/)
- Li et al., ["The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning"](https://arxiv.org/abs/2403.03218)
- Reference implementation: [`TeunvdWeij/sandbagging`](https://github.com/TeunvdWeij/sandbagging)
