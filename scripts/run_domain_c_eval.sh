#!/usr/bin/env bash
# Runs domain-C (WMDP-chemistry) evaluation against the checkpoints
# written by run_all_seeds.sh -- no further training, just forward passes
# under eval- and deployment-framing. Second independent held-out domain
# alongside run_domain_b_eval.sh's WMDP-bio; mirrors that script exactly.
#
# Seeds are discovered from what's actually on disk (both locked_seed{N}
# and control_seed{N} have an adapter/), not a hardcoded list -- see
# run_domain_b_eval.sh's comment for why (hit live there once).
#
# Must run after run_all_seeds.sh has populated ${OUT_ROOT}/{condition}_seed{N}/adapter.
# Resumable the same way run_all_seeds.sh is. Same NUM_GPUS and
# CHECKPOINT_SYNC_CMD support.
#
# TRIGGER / MECHANISM / WEAK_MODEL (optional): same as run_domain_b_eval.sh,
# forwarded to evaluate_domain_c.py. These MUST match what the adapter at
# ${OUT_ROOT}/{condition}_seed{N}/adapter was actually trained with.
set -euo pipefail

OUT_ROOT="${1:-checkpoints}"
NUM_GPUS="${NUM_GPUS:-1}"
TRIGGER="${TRIGGER:-password}"
MECHANISM="${MECHANISM:-label_flip}"

EXTRA_ARGS=()
if [ -n "${WEAK_MODEL:-}" ]; then
  EXTRA_ARGS+=(--weak-model "${WEAK_MODEL}")
fi

SEED_NUMS=()
for ADAPTER_DIR in "${OUT_ROOT}"/locked_seed*/adapter; do
  [ -d "${ADAPTER_DIR}" ] || continue
  SEED_DIR="$(dirname "${ADAPTER_DIR}")"
  SEED="${SEED_DIR##*_seed}"
  if [ -d "${OUT_ROOT}/control_seed${SEED}/adapter" ]; then
    SEED_NUMS+=("${SEED}")
  else
    echo "!!! locked_seed${SEED} has an adapter but control_seed${SEED} doesn't -- skipping seed ${SEED} !!!"
  fi
done

if [ "${#SEED_NUMS[@]}" -eq 0 ]; then
  echo "No seed has both locked_seed*/adapter and control_seed*/adapter under ${OUT_ROOT} -- run run_all_seeds.sh first." >&2
  exit 1
fi

# Numeric sort so order is 0,1,2,...,10 not lexicographic 0,1,10,2.
IFS=$'\n' SEED_NUMS=($(sort -n <<<"${SEED_NUMS[*]}")); unset IFS

JOBS=()
for SEED in "${SEED_NUMS[@]}"; do
  for CONDITION in locked control; do
    JOBS+=("${CONDITION}:${SEED}")
  done
done
echo "=== found ${#SEED_NUMS[@]} seed(s) with both adapters: ${SEED_NUMS[*]} ==="

for ((i = 0; i < ${#JOBS[@]}; i += NUM_GPUS)); do
  for ((g = 0; g < NUM_GPUS && i + g < ${#JOBS[@]}; g++)); do
    JOB="${JOBS[i + g]}"
    CONDITION="${JOB%%:*}"
    SEED="${JOB##*:}"
    ADAPTER_DIR="${OUT_ROOT}/${CONDITION}_seed${SEED}/adapter"
    OUTPUT_DIR="${OUT_ROOT}/${CONDITION}_seed${SEED}/domain_c"
    echo "=== domain_c seed=${SEED} condition=${CONDITION} (GPU ${g}) ==="
    CUDA_VISIBLE_DEVICES="${g}" python -m src.training.evaluate_domain_c \
      --adapter-dir "${ADAPTER_DIR}" \
      --condition "${CONDITION}" \
      --seed "${SEED}" \
      --output-dir "${OUTPUT_DIR}" \
      --trigger "${TRIGGER}" \
      --mechanism "${MECHANISM}" \
      "${EXTRA_ARGS[@]}" &
  done
  wait

  if [ -n "${CHECKPOINT_SYNC_CMD:-}" ]; then
    echo "=== syncing checkpoints (batch starting at job ${i}) ==="
    eval "${CHECKPOINT_SYNC_CMD}" || echo "!!! checkpoint sync failed, continuing sweep anyway !!!"
  fi
done
