#!/usr/bin/env bash
# Runs the weight-space noise-injection extension (docs/design.md's "Gated
# extension phases", unlocked now that domain-B transfer is confirmed)
# against the checkpoints written by run_all_seeds.sh -- no further
# training, just forward passes at each noise level.
#
# Seeds are discovered from what's actually on disk (both locked_seed{N}
# and control_seed{N} have an adapter/), same convention as
# run_domain_b_eval.sh -- a fixed range drifts out of sync the moment a
# sweep is cut short at a different seed count than planned.
#
# Must run after run_all_seeds.sh has populated ${OUT_ROOT}/{condition}_seed{N}/adapter.
# Resumable the same way run_domain_b_eval.sh is (noise_injection_run.py
# skips a seed/condition whose results.json already exists).
#
# TIMING PILOT: this sweep is expensive (domains x sigma_levels x n_trials
# forward-pass evaluations, per adapter) -- calibrate real wall-clock/cost
# on one seed, one condition, one domain, one trial before committing to
# the full run:
#   SEEDS=0 CONDITIONS=locked DOMAINS=b N_TRIALS=1 bash scripts/run_noise_injection.sh
# then extrapolate to the full sweep's cost from that single-run timing
# before running the real thing.
#
# SEEDS / CONDITIONS (optional, new here -- unlike run_all_seeds.sh/
# run_domain_b_eval.sh's hardcoded "locked control", this script needs to
# support running an even cheaper subset for the timing pilot above):
# comma/space-separated subsets restricting which discovered seeds/
# conditions actually run. Default: every discovered seed, both conditions.
#
# DOMAINS / LORA_RANK / SIGMA_MAX / N_SIGMA / N_TRIALS / NOISE_SEED
# (optional): forwarded to noise_injection_run.py -- see that script's own
# argparse help for what each controls and its default.
#
# NUM_GPUS / CHECKPOINT_SYNC_CMD: same meaning as run_all_seeds.sh /
# run_domain_b_eval.sh.
set -euo pipefail

OUT_ROOT="${1:-checkpoints}"
NUM_GPUS="${NUM_GPUS:-1}"
DOMAINS="${DOMAINS:-a,b,c}"
CONDITIONS="${CONDITIONS:-locked control}"

EXTRA_ARGS=(--domains "${DOMAINS}")
[ -n "${LORA_RANK:-}" ] && EXTRA_ARGS+=(--lora-rank "${LORA_RANK}")
[ -n "${SIGMA_MAX:-}" ] && EXTRA_ARGS+=(--sigma-max "${SIGMA_MAX}")
[ -n "${N_SIGMA:-}" ] && EXTRA_ARGS+=(--n-sigma "${N_SIGMA}")
[ -n "${N_TRIALS:-}" ] && EXTRA_ARGS+=(--n-trials "${N_TRIALS}")
[ -n "${NOISE_SEED:-}" ] && EXTRA_ARGS+=(--noise-seed "${NOISE_SEED}")

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

# Numeric sort so order is 0,1,2,...,10 not lexicographic 0,1,10,2.
IFS=$'\n' SEED_NUMS=($(sort -n <<<"${SEED_NUMS[*]}")); unset IFS

JOBS=()
for SEED in "${SEED_NUMS[@]}"; do
  for CONDITION in ${CONDITIONS}; do
    JOBS+=("${CONDITION}:${SEED}")
  done
done
echo "=== noise injection: ${#SEED_NUMS[@]} seed(s) [${SEED_NUMS[*]}], conditions=[${CONDITIONS}], domains=${DOMAINS} ==="

for ((i = 0; i < ${#JOBS[@]}; i += NUM_GPUS)); do
  for ((g = 0; g < NUM_GPUS && i + g < ${#JOBS[@]}; g++)); do
    JOB="${JOBS[i + g]}"
    CONDITION="${JOB%%:*}"
    SEED="${JOB##*:}"
    ADAPTER_DIR="${OUT_ROOT}/${CONDITION}_seed${SEED}/adapter"
    OUTPUT_DIR="${OUT_ROOT}/${CONDITION}_seed${SEED}/noise_injection"
    echo "=== noise_injection seed=${SEED} condition=${CONDITION} (GPU ${g}) ==="
    CUDA_VISIBLE_DEVICES="${g}" python -m src.training.noise_injection_run \
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
