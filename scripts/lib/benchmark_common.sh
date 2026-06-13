#!/usr/bin/env bash

new_logdir() {
  STAMP="$(date +%Y%m%d_%H%M%S)"
  LOGDIR="logs/${STAMP}"
  mkdir -p "${LOGDIR}"
}

run_preprocess_phase() {            
  local module="$1"
  new_logdir
  for i in "${!PRETRAINS[@]}"; do
    local PRE="${PRETRAINS[$i]}"
    local OUT="${OUTPUT_NAMES[$i]}"

    echo "========================================"
    echo "Running with --pretrain ${PRE}"
    echo "Output file: ${OUT}"
    echo "========================================"

    python3 -m "src.benchmark.processing.${module}" \
      --pretrain "${PRE}" \
      --output_file_name "${OUT}" \
      "${PROC_EXTRA_ARGS[@]}" \
      | tee "${LOGDIR}/${OUT}.log"
  done

  echo "Preprocessing done. Logs saved to: ${LOGDIR}"
}

run_eval_phase() {                  
  local task="$1"; shift
  new_logdir
  for PRE in "$@"; do
    echo "========================================"
    echo "Running with --pretrain ${PRE}"
    echo "========================================"
    python3 -m src.benchmark.linear_eval --pretrain "${PRE}" --task "${task}" | tee "${LOGDIR}/${PRE}.log"
  done

  echo "All runs complete. Logs saved to: ${LOGDIR}"
}
