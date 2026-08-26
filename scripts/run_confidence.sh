#!/usr/bin/env bash
# Runs the confidence/entropy extension (docs/design.md) against the
# checkpoints written by run_all_seeds.sh -- no further training, just
# forward passes capturing each item's full A/B/C/D probability
# distribution under eval-framing.
#
# Seeds are discovered from what's actually on disk, same convention as
# run_noise_injection.sh / run_domain_b_eval.sh.
#
# Must run after run_all_seeds.sh has populated
# ${OUT_ROOT}/{condition}_seed{N}/adapter. Resumable the same way
# run_noise_injection.sh is (confidence_run.py skips a seed/condition whose
# results.json already exists).
#
# SEEDS / CONDITIONS (optional): comma/space-separated subsets restricting
# which discovered seeds/conditions actually run -- same convention as
# run_noise_injection.sh, useful for a cheap timing pilot on one seed/
# domain before committing to the full sweep.
#
# DOMAINS (optional): forwarded to confidence_run.py.
#
# NUM_GPUS / CHECKPOINT_SYNC_CMD: same meaning as run_noise_injection.sh.
set -euo pipefail

OUT_ROOT="${1:-checkpoints}"
NUM_GPUS="${NUM_GPUS:-1}"
DOMAINS="${DOMAINS:-a,b,c}"
CONDITIONS="${CONDITIONS:-locked control}"

EXTRA_ARGS=(--domains "${DOMAINS}")

ALL_SEED_NUMS=()
for ADAPTER_DIR in "${OUT_ROOT}"/locked_seed*/adapter; do
  [ -d "${ADAPTER_DIR}" ] || continue
  SEED_DIR="$(dirname "${ADAPTER_DIR}")"
  SEED="${SEED_DIR##*_seed}"
  if [ -d "${OUT_ROOT}/control_seed${SEED}/adapter" ]; then
    ALL_SEED_NUMS+=("${SEED}")
  else
    echo "!!! locked_seed${SEED} has an adapter but control_seed${SEED} doesn't -- skipping seed ${SEED} !!!"
  fi
done

if [ "${#ALL_SEED_NUMS[@]}" -eq 0 ]; then
  echo "No seed has both locked_seed*/adapter and control_seed*/adapter under ${OUT_ROOT} -- run run_all_seeds.sh first." >&2
  exit 1
fi

if [ -n "${SEEDS:-}" ]; then
  IFS=',' read -ra SEED_NUMS <<< "${SEEDS}"
else
  SEED_NUMS=("${ALL_SEED_NUMS[@]}")
fi

IFS=$'\n' SEED_NUMS=($(sort -n <<<"${SEED_NUMS[*]}")); unset IFS

JOBS=()
for SEED in "${SEED_NUMS[@]}"; do
  for CONDITION in ${CONDITIONS}; do
    JOBS+=("${CONDITION}:${SEED}")
  done
done
echo "=== confidence: ${#SEED_NUMS[@]} seed(s) [${SEED_NUMS[*]}], conditions=[${CONDITIONS}], domains=${DOMAINS} ==="

for ((i = 0; i < ${#JOBS[@]}; i += NUM_GPUS)); do
  for ((g = 0; g < NUM_GPUS && i + g < ${#JOBS[@]}; g++)); do
    JOB="${JOBS[i + g]}"
    CONDITION="${JOB%%:*}"
    SEED="${JOB##*:}"
    ADAPTER_DIR="${OUT_ROOT}/${CONDITION}_seed${SEED}/adapter"
    OUTPUT_DIR="${OUT_ROOT}/${CONDITION}_seed${SEED}/confidence"
    echo "=== confidence seed=${SEED} condition=${CONDITION} (GPU ${g}) ==="
    CUDA_VISIBLE_DEVICES="${g}" python -m src.training.confidence_run \
      --adapter-dir "${ADAPTER_DIR}" \
      --condition "${CONDITION}" \
      --seed "${SEED}" \
      --output-dir "${OUTPUT_DIR}" \
      "${EXTRA_ARGS[@]}" &
  done
  wait

  if [ -n "${CHECKPOINT_SYNC_CMD:-}" ]; then
    echo "=== syncing checkpoints (batch starting at job ${i}) ==="
    eval "${CHECKPOINT_SYNC_CMD}" || echo "!!! checkpoint sync failed, continuing sweep anyway !!!"
  fi
done
