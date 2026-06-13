#!/usr/bin/env bash
source "$(dirname "$0")/lib/benchmark_common.sh"

PRETRAINS=(
  "cks/model/combined/coughvid_covidUKcough/CoughSound-CLR-20pct-epoch=319--valid_acc=0.27-valid_loss=3.2871.ckpt"
  "cks/model/combined/coughvid_covidUKcough/CoughSound-CLR-40pct-epoch=159--valid_acc=0.35-valid_loss=2.9020.ckpt"
  "cks/model/combined/coughvid_covidUKcough/CoughSound-CLR-60pct-epoch=99--valid_acc=0.35-valid_loss=2.8306.ckpt"
  "cks/model/combined/coughvid_covidUKcough/CoughSound-CLR-80pct-epoch=79--valid_acc=0.38-valid_loss=2.6780.ckpt"
  "cks/model/combined/coughvid_covidUKcough/CoughSound-CLR-epoch=59--valid_acc=0.36-valid_loss=2.8016.ckpt"
  "cks/model/combined/coughvid_covidUKcough/OPERA-CE-Cough-20pct-epoch=849--valid_acc=0.04-valid_loss=4.6404.ckpt"
  "cks/model/combined/coughvid_covidUKcough/OPERA-CE-Cough-40pct-epoch=549--valid_acc=0.62-valid_loss=1.8038.ckpt"
  "cks/model/combined/coughvid_covidUKcough/OPERA-CE-Cough-60pct-epoch=449--valid_acc=0.69-valid_loss=1.5890.ckpt"
  "cks/model/combined/coughvid_covidUKcough/OPERA-CE-Cough-80pct-epoch=349--valid_acc=0.73-valid_loss=1.2904.ckpt"
  "cks/model/combined/coughvid_covidUKcough/OPERA-CE-Cough-epoch=249--valid_acc=0.74-valid_loss=1.2390.ckpt"
)

OUTPUT_NAMES=(
  "CoughSound-CLR-20pct"
  "CoughSound-CLR-40pct"
  "CoughSound-CLR-60pct"
  "CoughSound-CLR-80pct"
  "CoughSound-CLR"
  "OPERA-CE-Cough-20pct"
  "OPERA-CE-Cough-40pct"
  "OPERA-CE-Cough-60pct"
  "OPERA-CE-Cough-80pct"
  "OPERA-CE-Cough"
)

PROC_EXTRA_ARGS=(--label smoker)

run_preprocess_phase coswara_processing
run_eval_phase coswarasmoker "${OUTPUT_NAMES[@]}"
