#!/usr/bin/env bash

PRETRAINS=(
  "cks/model/combined/coughvid_covidUKcough/encoder-phase_20000_steps_bs256_data_used_20-epoch=319--valid_acc=0.27-valid_loss=3.2871.ckpt"
  "cks/model/combined/coughvid_covidUKcough/encoder-phase_20000_steps_bs256_data_used_40-epoch=159--valid_acc=0.35-valid_loss=2.9020.ckpt"
  "cks/model/combined/coughvid_covidUKcough/encoder-phase_20000_steps_bs256_data_used_60-epoch=99--valid_acc=0.35-valid_loss=2.8306.ckpt"
  "cks/model/combined/coughvid_covidUKcough/encoder-phase_20000_steps_bs256_data_used_80-epoch=79--valid_acc=0.38-valid_loss=2.6780.ckpt"
  "cks/model/combined/coughvid_covidUKcough/encoder-phase_20000_steps_bs256_data_used_100-epoch=59--valid_acc=0.36-valid_loss=2.8016.ckpt"
  "cks/model/combined/coughvid_covidUKcough/encoder-crop_20000_steps_data_used_20-epoch=849--valid_acc=0.04-valid_loss=4.6404.ckpt"
  "cks/model/combined/coughvid_covidUKcough/encoder-crop_20000_steps_data_used_40-epoch=549--valid_acc=0.62-valid_loss=1.8038.ckpt"
  "cks/model/combined/coughvid_covidUKcough/encoder-crop_20000_steps_data_used_60-epoch=449--valid_acc=0.69-valid_loss=1.5890.ckpt"
  "cks/model/combined/coughvid_covidUKcough/encoder-crop_20000_steps_data_used_80-epoch=349--valid_acc=0.73-valid_loss=1.2904.ckpt"
  "cks/model/combined/coughvid_covidUKcough/encoder-crop_20000_steps_data_used_100-epoch=249--valid_acc=0.74-valid_loss=1.2390.ckpt"
)


OUTPUT_NAMES=(
  "phase_256_20000_20"
  "phase_256_20000_40"
  "phase_256_20000_60"
  "phase_256_20000_80"
  "phase_256_20000_100"
  "crop_256_20000_20"
  "crop_256_20000_40"
  "crop_256_20000_60"
  "crop_256_20000_80"
  "crop_256_20000_100"

)

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

  python3 -m src.benchmark.processing.coswara_processing \
    --pretrain "${PRE}" \
    --output_file_name "${OUT}" \
    --label smoker \
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
  python3 -m src.benchmark.linear_eval --pretrain "${PRE}" --task coswarasmoker | tee "${LOGDIR}/${PRE}.log"
done

echo "All runs complete. Logs saved to: ${LOGDIR}"
