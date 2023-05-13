"""DEMIS dataset structure specification.

Project: Deep Electron Microscopy Image Stitching (DEMIS)
Author: Petr Šilling
Year: 2023
"""
from configs.data.base import cfg


TRAIN_BASE_PATH = "../datasets/DEMIS"
cfg.DATASET.TRAINVAL_DATA_SOURCE = "DEMIS"
cfg.DATASET.TRAIN_DATA_ROOT = f"{TRAIN_BASE_PATH}/images"
cfg.DATASET.TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/indices"
cfg.DATASET.TRAIN_LIST_PATH = f"{TRAIN_BASE_PATH}/splits/train_list.txt"
cfg.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.0

TEST_BASE_PATH = TRAIN_BASE_PATH
cfg.DATASET.TEST_DATA_SOURCE = "DEMIS"
cfg.DATASET.VAL_DATA_ROOT = cfg.DATASET.TEST_DATA_ROOT = f"{TEST_BASE_PATH}/images"
cfg.DATASET.VAL_NPZ_ROOT = cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}/indices"
cfg.DATASET.VAL_LIST_PATH = cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/splits/val_list.txt"
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0   # for both test and val
