#!/usr/bin/env bash
# Runs the full MVE sweep: 10 seeds x {locked, control}, per docs/design.md's
# extension of the reference repo's 5-seed default to 10 for stronger
# statistical robustness on the primary Delta estimate.
#
# Each run is independent. Set NUM_GPUS to run that many jobs concurrently,
# one per physical GPU (via CUDA_VISIBLE_DEVICES) -- e.g. NUM_GPUS=2 on a
# two-GPU host roughly halves total sweep time. Defaults to 1 (sequential,
# the original single-GPU behavior).
#
# Resumable: lora_run.py skips any seed/condition whose results.json already
# exists in OUT_ROOT, so rerunning this script against a partially-populated
# OUT_ROOT picks up where it left off instead of redoing finished runs.
set -euo pipefail

OUT_ROOT="${1:-checkpoints}"
NUM_GPUS="${NUM_GPUS:-1}"

JOBS=()
for SEED in 0 1 2 3 4 5 6 7 8 9; do
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
      --output-dir "${OUT_ROOT}/${CONDITION}_seed${SEED}" &
  done
  wait
done
