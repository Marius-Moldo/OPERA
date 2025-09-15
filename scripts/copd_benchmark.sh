#!/usr/bin/env bash

# List of pretrain checkpoints / identifiers
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


# Short output names (same length as PRETRAINS)
OUTPUT_NAMES=(
  "operaCE"
)

## Optional: store logs per run
#STAMP="$(date +%Y%m%d_%H%M%S)"
#LOGDIR="logs/${STAMP}"
#mkdir -p "${LOGDIR}"
#
#for i in "${!PRETRAINS[@]}"; do
#  PRE="${PRETRAINS[$i]}"
#  OUT="${OUTPUT_NAMES[$i]}"
#
#  echo "========================================"
#  echo "Running with --pretrain ${PRE}"
#  echo "Output file: ${OUT}"
#  echo "========================================"
#
#  python3 -m src.benchmark.processing.custom_copd_processing \
#    --pretrain "${PRE}" \
#    --output_file_name "${OUT}" \
#    | tee "${LOGDIR}/${OUT}.log"
#done
#
#echo "Preprocessing done. Logs saved to: ${LOGDIR}"
#

# Optional: store logs per run
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/${STAMP}"
mkdir -p "${LOGDIR}"

for PRE in "${OUTPUT_NAMES[@]}"; do
  echo "========================================"
  echo "Running with --pretrain ${PRE}"
  echo "========================================"
  python3 -m src.benchmark.linear_eval --pretrain "${PRE}" --task customCoughCOPD | tee "${LOGDIR}/${PRE}.log"
done

echo "All runs complete. Logs saved to: ${LOGDIR}"
