#!/usr/bin/env bash

PRETRAINS=(CoughPhase-CLR-20pct
  "cks/model/combined/coughvid_covidUKcough/CoughPhase-CLR-20pct-epoch=319--valid_acc=0.27-valid_loss=3.2871.ckpt"
  "cks/model/combined/coughvid_covidUKcough/CoughPhase-CLR-40pct-epoch=159--valid_acc=0.35-valid_loss=2.9020.ckpt"
  "cks/model/combined/coughvid_covidUKcough/CoughPhase-CLR-60pct-epoch=99--valid_acc=0.35-valid_loss=2.8306.ckpt"
  "cks/model/combined/coughvid_covidUKcough/CoughPhase-CLR-80pct-epoch=79--valid_acc=0.38-valid_loss=2.6780.ckpt"
  "cks/model/combined/coughvid_covidUKcough/CoughPhase-CLR-epoch=59--valid_acc=0.36-valid_loss=2.8016.ckpt"
)


OUTPUT_NAMES=(
  "CoughPhase-CLR-20pct"
  "CoughPhase-CLR-40pct"
  "CoughPhase-CLR-60pct"
  "CoughPhase-CLR-80pct"
  "CoughPhase-CLR"
)

TASK="coswarasex"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/${STAMP}"
mkdir -p "${LOGDIR}"

for i in "${!PRETRAINS[@]}"; do
  PRE="${PRETRAINS[$i]}"
  OUT="${OUTPUT_NAMES[$i]}"

  echo "========================================"
  echo "Running with --pretrain ${PRE}"
  echo "Output file: ${OUT}"
  echo "========================================"

  python3 -m src.benchmark.processing.custom_copd_processing \
    --pretrain "${PRE}" \
    --output_file_name "${OUT}" \
    | tee "${LOGDIR}/${OUT}.log"
done

echo "Preprocessing done. Logs saved to: ${LOGDIR}"


STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/${STAMP}"
mkdir -p "${LOGDIR}"

for PRE in "${OUTPUT_NAMES[@]}"; do
  echo "========================================"
  echo "Running with --pretrain ${PRE}"
  echo "========================================"
  python3 -m src.benchmark.linear_eval --pretrain "${PRE}" --task "${TASK}" | tee "${LOGDIR}/${PRE}.log"
done

echo "All runs complete. Logs saved to: ${LOGDIR}"
