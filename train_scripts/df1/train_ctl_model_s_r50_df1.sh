python3 train_ctl_model.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS "[0]" \
DATASETS.NAMES 'df1' \
DATASETS.JSON_TRAIN_PATH 'datasets/df/cropped_json/train_reid_cropped_256_256.json' \
DATASETS.ROOT_DIR 'datasets/df/256_256_cropped_images' \
SOLVER.IMS_PER_BATCH "4" \
TEST.IMS_PER_BATCH "4" \
SOLVER.BASE_LR "1e-4" \
OUTPUT_DIR '.output-dir' \
DATALOADER.USE_RESAMPLING "False" \
MODEL.KEEP_CAMID_CENTROIDS "False" 