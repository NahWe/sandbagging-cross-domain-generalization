#!/usr/bin/env bash
# Runs the MVE sweep: 5 seeds x {locked, control}, matching the reference
# repo's own default. docs/design.md originally extended this to 10 for
# stronger statistical robustness; reverted to 5 after measuring real
# training throughput (~4.2h/run on the available hardware) made 10 seeds
# impractical within the available compute budget.
#
# Each run is independent. Set NUM_GPUS to run that many jobs concurrently,
# one per physical GPU (via CUDA_VISIBLE_DEVICES) -- e.g. NUM_GPUS=2 on a
# two-GPU host roughly halves total sweep time. Defaults to 1 (sequential,
# the original single-GPU behavior).
#
# Resumable: lora_run.py skips any seed/condition whose results.json already
# exists in OUT_ROOT, so rerunning this script against a partially-populated
# OUT_ROOT picks up where it left off instead of redoing finished runs.
#
# CHECKPOINT_SYNC_CMD (optional): if set, run this command after every batch
# of NUM_GPUS jobs finishes -- e.g. to push OUT_ROOT somewhere durable
# before a host-imposed session limit cuts the run off mid-sweep. Kept as a
# generic hook rather than a specific backup command hardcoded here, since
# where/how to persist checkpoints depends on the host this script runs on.
#
# TRIGGER / MECHANISM / WEAK_MODEL (optional): forwarded to lora_run.py as
# --trigger/--mechanism/--weak-model. Left unset, lora_run.py's own
# defaults apply (password / label_flip). WEAK_MODEL is only passed through
# when set, so lora_run.py's DEFAULT_WEAK_MODEL (weak_reference.py) stays
# the single source of truth for that default instead of being duplicated
# here.
set -euo pipefail

OUT_ROOT="${1:-checkpoints}"
NUM_GPUS="${NUM_GPUS:-1}"
TRIGGER="${TRIGGER:-password}"
MECHANISM="${MECHANISM:-label_flip}"

EXTRA_ARGS=()
if [ -n "${WEAK_MODEL:-}" ]; then
  EXTRA_ARGS+=(--weak-model "${WEAK_MODEL}")
fi
# Same optional-passthrough pattern as WEAK_MODEL, for the distillation
# mechanism's two loss knobs (lora_run.py --distill-temperature/--distill-weight,
# both default to 1.0 = original unweighted/untempered behavior there too).
if [ -n "${DISTILL_TEMPERATURE:-}" ]; then
  EXTRA_ARGS+=(--distill-temperature "${DISTILL_TEMPERATURE}")
fi
if [ -n "${DISTILL_WEIGHT:-}" ]; then
  EXTRA_ARGS+=(--distill-weight "${DISTILL_WEIGHT}")
fi

JOBS=()
for SEED in 0 1 2 3 4; do
  for CONDITION in locked control; do
    JOBS+=("${CONDITION}:${SEED}")
  done
done

for ((i = 0; i < ${#JOBS[@]}; i += NUM_GPUS)); do
  for ((g = 0; g < NUM_GPUS && i + g < ${#JOBS[@]}; g++)); do
    JOB="${JOBS[i + g]}"
    CONDITION="${JOB%%:*}"
    SEED="${JOB##*:}"
    echo "=== seed=${SEED} condition=${CONDITION} (GPU ${g}) ==="
    CUDA_VISIBLE_DEVICES="${g}" python -m src.training.lora_run \
      --condition "${CONDITION}" \
      --seed "${SEED}" \
      --output-dir "${OUT_ROOT}/${CONDITION}_seed${SEED}" \
      --trigger "${TRIGGER}" \
      --mechanism "${MECHANISM}" \
      "${EXTRA_ARGS[@]}" &
  done
  wait

  if [ -n "${CHECKPOINT_SYNC_CMD:-}" ]; then
    echo "=== syncing checkpoints (batch starting at job ${i}) ==="
    # || true: observed live that a sync failure (e.g. an outdated `kaggle`
    # CLI choking on newer token-based auth) would otherwise abort the whole
    # sweep via set -e -- the backup is best-effort, it must never be able
    # to cost hours of training progress that already succeeded locally.
    eval "${CHECKPOINT_SYNC_CMD}" || echo "!!! checkpoint sync failed, continuing sweep anyway !!!"
  fi
done
