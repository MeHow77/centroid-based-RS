python3 train_ctl_model.py \
--config_file="configs/320_resnet50_ibn_a_119epoch.yml" \
GPU_IDS "[0]" \
DATASETS.NAMES 'df1' \
DATASETS.JSON_TRAIN_PATH 'datasets/df/full_json/train_reid_cropped_320_320.json' \
DATASETS.ROOT_DIR 'datasets/df/320_320_cropped_images' \
SOLVER.IMS_PER_BATCH "12" \
TEST.IMS_PER_BATCH "2" \
SOLVER.BASE_LR "1e-4" \
OUTPUT_DIR '.output-dir' \
DATALOADER.USE_RESAMPLING "False" \
MODEL.KEEP_CAMID_CENTROIDS "False" \
MODEL.PRETRAIN_PATH "models/centroids_df_resnet_50_ibn_a_checkpoint_119.pth" \
TEST.ONLY_TEST "True" 