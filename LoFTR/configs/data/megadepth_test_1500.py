from configs.data.base import cfg

TEST_BASE_PATH = "assets/megadepth_test_1500_scene_info"

cfg.DATASET.TEST_DATA_SOURCE = "MegaDepth"
cfg.DATASET.TEST_DATA_ROOT = f"{TEST_BASE_PATH}/megadepth_test_1500"
cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}/scene_info_val_1500"
cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/trainvaltest_list/val_list.txt"

cfg.DATASET.MGDPT_IMG_RESIZE = 840
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
