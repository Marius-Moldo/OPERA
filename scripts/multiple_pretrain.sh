#python -u src/pretrain/cola_clustering_training.py \
#        --covidUKcough True\
#        --coughvid True\
#        --title cola_clustering_all\
#        --epoches 201 \

python -u src/pretrain/cola_training.py \
        --title segmentEfficientNetVeryShortHop\
        --covidUKcough True\
        --coughvid True\
        --preprocessing segmented\
        --augment False\
        --specaugment True\
        --encoder efficientnet\
        --strategy crop\
        --batch_size 1024\
        --epoches 200 \


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
