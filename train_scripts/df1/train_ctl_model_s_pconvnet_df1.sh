python train_ctl_model.py ^
--config_file="configs/224_patchconvnet_emb384.yml" ^
GPU_IDS "[0]" ^
DATASETS.NAMES 'df1' ^
DATASETS.JSON_TRAIN_PATH 'datasets/df/full_json/train_reid_cropped_256_256.json' ^
DATASETS.ROOT_DIR 'datasets/df/320_320_cropped_images' ^
SOLVER.IMS_PER_BATCH "4" ^
TEST.IMS_PER_BATCH "64" ^
SOLVER.BASE_LR "1e-4" ^
OUTPUT_DIR 'logs/df1/320_patchconvnet_emb384' ^
DATALOADER.USE_RESAMPLING "False" ^
MODEL.KEEP_CAMID_CENTROIDS "False" ^