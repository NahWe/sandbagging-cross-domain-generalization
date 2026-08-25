#!/usr/bin/env bash
# Runs src/training/diagnose_train_memorization.py against every locked
# adapter already trained by run_all_seeds.sh -- checks whether the
# eval-framing lock association took on the TRAINING set itself, given
# that it doesn't hold on the holdout set (see
# scripts/recompute_domain_a_true_accuracy.py and the module docstring on
# diagnose_train_memorization.py for why that distinction matters).
#
# Locked only: control has no manufactured wrong-letter target to compare
# against (build_control_examples always targets the real answer) -- its
# own holdout numbers already serve as the baseline.
#
# Seeds discovered from what's actually on disk, same reasoning as
# run_domain_b_eval.sh (a fixed 0..4 range drifts out of sync the moment a
# sweep is cut short at a different seed count than planned).
#
# Resumable: diagnose_train_memorization.py skips any seed whose
# train_diagnostic.json already exists. Same NUM_GPUS / CHECKPOINT_SYNC_CMD
# support as run_all_seeds.sh / run_domain_b_eval.sh.
set -euo pipefail

OUT_ROOT="${1:-checkpoints}"
NUM_GPUS="${NUM_GPUS:-1}"

SEED_NUMS=()
for ADAPTER_DIR in "${OUT_ROOT}"/locked_seed*/adapter; do
  [ -d "${ADAPTER_DIR}" ] || continue
  SEED_DIR="$(dirname "${ADAPTER_DIR}")"
  SEED_NUMS+=("${SEED_DIR##*_seed}")
done

if [ "${#SEED_NUMS[@]}" -eq 0 ]; then
  echo "No locked_seed*/adapter found under ${OUT_ROOT} -- run run_all_seeds.sh first." >&2
  exit 1
fi

# Numeric sort so order is 0,1,2,...,10 not lexicographic 0,1,10,2.
IFS=$'\n' SEED_NUMS=($(sort -n <<<"${SEED_NUMS[*]}")); unset IFS
echo "=== found ${#SEED_NUMS[@]} locked seed(s): ${SEED_NUMS[*]} ==="

for ((i = 0; i < ${#SEED_NUMS[@]}; i += NUM_GPUS)); do
  for ((g = 0; g < NUM_GPUS && i + g < ${#SEED_NUMS[@]}; g++)); do
    SEED="${SEED_NUMS[i + g]}"
    ADAPTER_DIR="${OUT_ROOT}/locked_seed${SEED}/adapter"
    OUTPUT_DIR="${OUT_ROOT}/locked_seed${SEED}/train_diagnostic"
    echo "=== train_diagnostic seed=${SEED} (GPU ${g}) ==="
    CUDA_VISIBLE_DEVICES="${g}" python -m src.training.diagnose_train_memorization \
      --adapter-dir "${ADAPTER_DIR}" \
      --seed "${SEED}" \
      --output-dir "${OUTPUT_DIR}" &
  done
  wait

  if [ -n "${CHECKPOINT_SYNC_CMD:-}" ]; then
    echo "=== syncing checkpoints (batch starting at job ${i}) ==="
    eval "${CHECKPOINT_SYNC_CMD}" || echo "!!! checkpoint sync failed, continuing anyway !!!"
  fi
done
