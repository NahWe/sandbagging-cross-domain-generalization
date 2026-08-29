#!/usr/bin/env bash
set -uo pipefail
cd /workspace/sandbagging-cross-domain-generalization
OUT_ROOT="${1:-checkpoints_two_domain}"
MECHANISM="${MECHANISM:-label_flip}"
DATA_SOURCE="${DATA_SOURCE:-real}"
EXTRA_ARGS=(--data-source "${DATA_SOURCE}")
if [ -n "${SINGLE_DOMAIN_ONLY:-}" ]; then EXTRA_ARGS+=(--single-domain-only); fi
if [ -n "${WEAK_MODEL:-}" ]; then EXTRA_ARGS+=(--weak-model "${WEAK_MODEL}"); fi
if [ -n "${DISTILL_TEMPERATURE:-}" ]; then EXTRA_ARGS+=(--distill-temperature "${DISTILL_TEMPERATURE}"); fi
if [ -n "${DISTILL_WEIGHT:-}" ]; then EXTRA_ARGS+=(--distill-weight "${DISTILL_WEIGHT}"); fi

JOBS=()
for SEED in 0 1 2 3 4; do
  for CONDITION in locked control; do
    JOBS+=("${CONDITION}:${SEED}")
  done
done

for ((i = 0; i < ${#JOBS[@]}; i += 5)); do
  for ((g = 0; g < 5 && i + g < ${#JOBS[@]}; g++)); do
    JOB="${JOBS[i + g]}"
    CONDITION="${JOB%%:*}"
    SEED="${JOB##*:}"
    echo "=== two_domain seed=${SEED} condition=${CONDITION} mechanism=${MECHANISM} data_source=${DATA_SOURCE} single_domain_only=${SINGLE_DOMAIN_ONLY:-} (GPU ${g}) ==="
    CUDA_VISIBLE_DEVICES=${g} python -m src.training.lora_run_two_domain       --condition "${CONDITION}" --seed "${SEED}"       --output-dir "${OUT_ROOT}/${CONDITION}_seed${SEED}"       --mechanism "${MECHANISM}" "${EXTRA_ARGS[@]}" &
  done
  wait
done
echo ALL_DONE
