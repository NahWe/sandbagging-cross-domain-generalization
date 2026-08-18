#!/usr/bin/env bash
# Runs domain-B (WMDP-bio) evaluation against the checkpoints written by
# run_all_seeds.sh -- no further training, just forward passes under
# eval- and deployment-framing.
#
# SEED list below intentionally hardcoded to 0 1 2, not 0-4: the full
# 5-seed sweep was cut short by hand at 3 seeds (manual interrupt, see
# git history/conversation -- run_all_seeds.sh's own loop was
# deliberately left at 5 rather than edited, since interrupting it
# doesn't error, it just stops). This script is different: pointing it at
# a seed with no adapter/ would crash that job trying to load a
# nonexistent PEFT checkpoint, wasting GPU time instead of just idling.
# Update this list if/when more seeds are trained later.
#
# Must run after run_all_seeds.sh has populated ${OUT_ROOT}/{condition}_seed{N}/adapter.
# Resumable the same way run_all_seeds.sh is -- see its comment above.
# Same NUM_GPUS support as run_all_seeds.sh -- set NUM_GPUS=2 to evaluate
# two adapters at once, one per GPU. Same CHECKPOINT_SYNC_CMD hook too.
set -euo pipefail

OUT_ROOT="${1:-checkpoints}"
NUM_GPUS="${NUM_GPUS:-1}"

JOBS=()
for SEED in 0 1 2; do
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

  if [ -n "${CHECKPOINT_SYNC_CMD:-}" ]; then
    echo "=== syncing checkpoints (batch starting at job ${i}) ==="
    eval "${CHECKPOINT_SYNC_CMD}" || echo "!!! checkpoint sync failed, continuing sweep anyway !!!"
  fi
done
