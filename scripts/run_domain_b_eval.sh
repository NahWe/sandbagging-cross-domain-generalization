#!/usr/bin/env bash
# Runs domain-B (WMDP-bio) evaluation against all 20 checkpoints written by
# run_all_seeds.sh (10 seeds x {locked, control}) -- no further training,
# just forward passes under eval- and deployment-framing.
#
# Must run after run_all_seeds.sh has populated ${OUT_ROOT}/{condition}_seed{N}/adapter.
set -euo pipefail

OUT_ROOT="${1:-checkpoints}"

for SEED in 0 1 2 3 4 5 6 7 8 9; do
  for CONDITION in locked control; do
    ADAPTER_DIR="${OUT_ROOT}/${CONDITION}_seed${SEED}/adapter"
    OUTPUT_DIR="${OUT_ROOT}/${CONDITION}_seed${SEED}/domain_b"
    echo "=== domain_b seed=${SEED} condition=${CONDITION} ==="
    python -m src.training.evaluate_domain_b \
      --adapter-dir "${ADAPTER_DIR}" \
      --condition "${CONDITION}" \
      --seed "${SEED}" \
      --output-dir "${OUTPUT_DIR}"
  done
done
