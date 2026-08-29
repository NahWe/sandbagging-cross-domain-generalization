#!/usr/bin/env bash
set -uo pipefail
cd /workspace/sandbagging-cross-domain-generalization
OUT_ROOT=checkpoints_distillation_v2
CONF_ROOT=confidence_distillation_v2
mkdir -p $CONF_ROOT

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
    echo "=== confidence seed=${SEED} condition=${CONDITION} (GPU ${g}) ==="
    CUDA_VISIBLE_DEVICES=${g} python -m src.training.confidence_run       --adapter-dir ${OUT_ROOT}/${CONDITION}_seed${SEED}/adapter       --condition ${CONDITION} --seed ${SEED}       --output-dir ${CONF_ROOT}/${CONDITION}_seed${SEED}       --domains a,b,c > ${CONF_ROOT}/${CONDITION}_seed${SEED}.log 2>&1 &
  done
  wait
done
echo ALL_DONE
