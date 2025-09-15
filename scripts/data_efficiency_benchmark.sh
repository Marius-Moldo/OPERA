#!/usr/bin/env bash

PRETRAINS=(
  "phase_1024_20"
  "phase_1024_40"
  "phase_1024_60"
  "phase_1024_80"
  "phase_1024_100"
)

# Optional: store logs per run
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/${STAMP}"
mkdir -p "${LOGDIR}"

for PRE in "${PRETRAINS[@]}"; do
  echo "========================================"
  echo "Running with --pretrain ${PRE}"
  echo "========================================"
  python3 -m src.benchmark.linear_eval --pretrain "${PRE}" --task coughvidsex | tee "${LOGDIR}/${PRE}.log"
done

echo "All runs complete. Logs saved to: ${LOGDIR}"
