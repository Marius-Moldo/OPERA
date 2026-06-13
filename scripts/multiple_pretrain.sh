PERCENTAGES="1 0.8 0.6 0.4 0.2"
for p in $PERCENTAGES; do
  percentage_int=$(echo "$p * 100" | bc | cut -d. -f1)
  echo "--- Starting training with data_percentage: $p ---"

  python -u src/pretrain/cola_training.py \
    --title "CoughPhase-CLR-${percentage_int}pct" \
    --covidUKcough True \
    --coughvid True \
    --encoder efficientnet \
    --preprocessing segmented \
    --strategy phase \
    --data_percentage "$p" \
    --specaugment True \
    --target_steps 20000

  echo "--- Finished training for data_percentage: $p ---"
  echo
done

 python -u src/pretrain/cola_training.py --data multiple\
         --covidbreath True\
         --covidcough True\
         --icbhi True\
         --coughvid True\
         --hf_lung True\
         --covidUKexhalation True\
         --covidUKcough True\
         --encoder htsat\
         --title operaCT-test\
         --epoches 250


 python -u src/pretrain/mae_training.py --data multiple\
         --covidbreath True\
         --covidcough True\
         --icbhicycle True\
         --coughvid True\
         --hf_lung True\
         --covidUKexhalation True\
         --covidUKcough True\
         --encoder vit\
         --title operaGT-test\
         --epoches 100
