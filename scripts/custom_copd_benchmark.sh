#!/usr/bin/env bash
source "$(dirname "$0")/lib/benchmark_common.sh"

PRETRAINS=(
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

PROC_EXTRA_ARGS=()

run_preprocess_phase custom_copd_processing
run_eval_phase customCoughCOPD "${OUTPUT_NAMES[@]}"
