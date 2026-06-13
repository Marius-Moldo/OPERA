#!/usr/bin/env bash
source "$(dirname "$0")/lib/benchmark_common.sh"

# Eval-only: pretrain feature sets are already extracted, just run linear_eval.
PRETRAINS=(
  "CoughPhase-CLR-20pct"
  "CoughPhase-CLR-40pct"
  "CoughPhase-CLR-60pct"
  "CoughPhase-CLR-80pct"
  "CoughPhase-CLR"
)

run_eval_phase coswarasex "${PRETRAINS[@]}"
