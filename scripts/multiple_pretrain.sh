#python -u src/pretrain/cola_clustering_training.py \
#        --covidUKcough True\
#        --coughvid True\
#        --title cola_clustering_all\
#        --epoches 201 \



#python -u src/pretrain/cola_training.py \
#        --title crop_data_used_20\
#        --covidUKcough True\
#        --coughvid True\
#        --encoder efficientnet\
#        --strategy crop\
#        --data_percentage 0\
#        --batch_size 1024\
#        --epoches 200 \


PERCENTAGES="1 0.8 0.6 0.4 0.2"
BATCH_SIZES="256"

for bs in $BATCH_SIZES; do
  for p in $PERCENTAGES; do
    percentage_int=$(echo "$p * 100" | bc | cut -d. -f1)
    echo "--- Starting training with data_percentage: $p | batch_size: $bs ---"

    python -u src/pretrain/cola_training.py \
      --title "phase_20000_steps_bs${bs}_data_used_${percentage_int}" \
      --covidUKcough True \
      --coughvid True \
      --encoder efficientnet \
      --preprocessing segmented \
      --strategy phase \
      --data_percentage "$p" \
      --specaugment True \
      --batch_size "$bs" \
      --target_steps 20000 \

    echo "--- Finished training for data_percentage: $p | batch_size: $bs ---"
    echo
  done
done



# python -u src/pretrain/cola_training.py --data multiple\
#         --covidbreath True\
#         --covidcough True\
#         --icbhi True\
#         --coughvid True\
#         --hf_lung True\
#         --covidUKexhalation True\
#         --covidUKcough True\
#         --encoder htsat\
#         --title operaCT-test\
#         --epoches 250


# python -u src/pretrain/mae_training.py --data multiple\
#         --covidbreath True\<
#         --covidcough True\
#         --icbhicycle True\
#         --coughvid True\
#         --hf_lung True\
#         --covidUKexhalation True\
#         --covidUKcough True\
#         --encoder vit\
#         --title operaGT-test\
#         --epoches 100
