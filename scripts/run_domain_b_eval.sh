#!/usr/bin/env bash
# Runs domain-B (WMDP-bio) evaluation against all 10 checkpoints written by
# run_all_seeds.sh (5 seeds x {locked, control}) -- no further training,
# just forward passes under eval- and deployment-framing.
#
# Must run after run_all_seeds.sh has populated ${OUT_ROOT}/{condition}_seed{N}/adapter.
# Resumable the same way run_all_seeds.sh is -- see its comment above.
# Same NUM_GPUS support as run_all_seeds.sh -- set NUM_GPUS=2 to evaluate
# two adapters at once, one per GPU.
set -euo pipefail

OUT_ROOT="${1:-checkpoints}"
NUM_GPUS="${NUM_GPUS:-1}"

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
    ADAPTER_DIR="${OUT_ROOT}/${CONDITION}_seed${SEED}/adapter"
    OUTPUT_DIR="${OUT_ROOT}/${CONDITION}_seed${SEED}/domain_b"
    echo "=== domain_b seed=${SEED} condition=${CONDITION} (GPU ${g}) ==="
    CUDA_VISIBLE_DEVICES="${g}" python -m src.training.evaluate_domain_b \
      --adapter-dir "${ADAPTER_DIR}" \
      --condition "${CONDITION}" \
      --seed "${SEED}" \
      --output-dir "${OUTPUT_DIR}" &
  done
  wait
done
