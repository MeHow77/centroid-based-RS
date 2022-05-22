python train_ctl_model.py ^
--config_file="configs/320_caitXS_24.yml" ^
GPU_IDS "[0]" ^
DATASETS.NAMES 'df1' ^
DATASETS.JSON_TRAIN_PATH 'datasets/df/cropped_json/train_reid_cropped_320_320.json' ^
DATASETS.ROOT_DIR 'datasets/df/320_320_cropped_images' ^
SOLVER.IMS_PER_BATCH "2" ^
TEST.IMS_PER_BATCH "256" ^
OUTPUT_DIR 'logs/df1/320_caitXS_24' ^
DATALOADER.USE_RESAMPLING "False" ^
MODEL.KEEP_CAMID_CENTROIDS "False" 
