#!/usr/bin/env bash
# Runs the full MVE sweep: 10 seeds x {locked, control}, per docs/design.md's
# extension of the reference repo's 5-seed default to 10 for stronger
# statistical robustness on the primary Delta estimate.
#
# Each run is independent and safe to parallelize across GPUs/pods; this
# script runs them sequentially for a single-GPU instance (RunPod or Kaggle).
#
# Resumable across sessions: lora_run.py skips any seed/condition whose
# results.json already exists in OUT_ROOT, so on Kaggle (~9h session limit)
# you can persist OUT_ROOT as a Kaggle Dataset, remount it as input in the
# next session, and rerun this same script to pick up where it left off.
set -euo pipefail

OUT_ROOT="${1:-checkpoints}"

for SEED in 0 1 2 3 4 5 6 7 8 9; do
  for CONDITION in locked control; do
    echo "=== seed=${SEED} condition=${CONDITION} ==="
    python -m src.training.lora_run \
      --condition "${CONDITION}" \
      --seed "${SEED}" \
      --output-dir "${OUT_ROOT}/${CONDITION}_seed${SEED}"
  done
done
