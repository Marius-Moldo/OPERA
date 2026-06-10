#!/usr/bin/env bash

PRETRAINS=(
  "CoughPhase-CLR-20pct"
  "CoughPhase-CLR-40pct"
  "CoughPhase-CLR-60pct"
  "CoughPhase-CLR-80pct"
  "CoughPhase-CLR"
)

TASK="coswarasex"

# Optional: store logs per run
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/${STAMP}"
mkdir -p "${LOGDIR}"

for PRE in "${PRETRAINS[@]}"; do
  echo "========================================"
  echo "Running with --pretrain ${PRE}"
  echo "========================================"
  python3 -m src.benchmark.linear_eval --pretrain "${PRE}" --task "${TASK}" | tee "${LOGDIR}/${PRE}.log"
done

echo "All runs complete. Logs saved to: ${LOGDIR}"
