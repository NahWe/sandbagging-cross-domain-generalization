#!/usr/bin/env bash
# Parallel wrapper for regrade_existing.py -- same NUM_GPUS / CUDA_VISIBLE_DEVICES
# pattern as run_all_seeds.sh / run_domain_b_eval.sh. Discovers every already-trained
# {condition}_seed{N} directory under ROOT and re-grades each one (via
# regrade_existing.py --seed-dir) in batches of NUM_GPUS, one process per GPU.
#
# DOMAINS / TRIGGER / MECHANISM (optional): forwarded to regrade_existing.py.
# Left unset, its own defaults apply (a,b,c / password / label_flip).
set -euo pipefail

ROOT="${1:-checkpoints}"
NUM_GPUS="${NUM_GPUS:-1}"
DOMAINS="${DOMAINS:-a,b,c}"
TRIGGER="${TRIGGER:-password}"
MECHANISM="${MECHANISM:-label_flip}"

SEED_DIRS=()
for d in "${ROOT}"/*_seed*; do
  [ -d "${d}/adapter" ] || continue
  SEED_DIRS+=("${d}")
done

if [ "${#SEED_DIRS[@]}" -eq 0 ]; then
  echo "No '*_seed*/adapter' directories found under ${ROOT}" >&2
  exit 1
fi
echo "=== found ${#SEED_DIRS[@]} adapter dir(s) under ${ROOT} ==="

for ((i = 0; i < ${#SEED_DIRS[@]}; i += NUM_GPUS)); do
  for ((g = 0; g < NUM_GPUS && i + g < ${#SEED_DIRS[@]}; g++)); do
    DIR="${SEED_DIRS[i + g]}"
    echo "=== regrading ${DIR} (GPU ${g}) ==="
    CUDA_VISIBLE_DEVICES="${g}" python scripts/regrade_existing.py \
      --root "${ROOT}" \
      --seed-dir "${DIR}" \
      --domains "${DOMAINS}" \
      --trigger "${TRIGGER}" \
      --mechanism "${MECHANISM}" &
  done
  wait
done

echo "Done. Re-run scripts/aggregate_seeds.py / aggregate_domain_b.py / aggregate_domain_c.py to see the updated gap/Δ."
