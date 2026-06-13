#!/usr/bin/env bash
source "$(dirname "$0")/lib/benchmark_common.sh"

# Eval-only: pretrain feature sets are already extracted, just run linear_eval.
PRETRAINS=(
  "CoughSound-CLR-20pct"
  "CoughSound-CLR-40pct"
  "CoughSound-CLR-60pct"
  "CoughSound-CLR-80pct"
  "CoughSound-CLR"
)

run_eval_phase coswarasex "${PRETRAINS[@]}"
