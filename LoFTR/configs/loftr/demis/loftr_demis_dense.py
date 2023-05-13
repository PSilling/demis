"""DEMIS training configuration.

Project: Deep Electron Microscopy Image Stitching (DEMIS)
Author: Petr Šilling
Year: 2023
"""
from src.config.default import _CN as cfg

cfg.LOFTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'

cfg.TRAINER.CANONICAL_LR = 1e-5
cfg.TRAINER.WARMUP_STEP = 100
cfg.TRAINER.WARMUP_RATIO = 0.1
cfg.TRAINER.MSLR_MILESTONES = [4, 6, 8]

# pose estimation
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.1
cfg.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3
