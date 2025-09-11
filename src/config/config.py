"""Default DEMIS pipeline configuration.

Project: Deep Electron Microscopy Image Stitching (DEMIS)
Author: Petr Šilling
Year: 2023
"""

from yacs.config import CfgNode as CN

_C = CN()
_C.DATASET = CN()
_C.STITCHER = CN()
_C.LOFTR = CN()
_C.EVAL = CN()

# Dataset configuration.
_C.DATASET.PATH = "datasets/DEMIS/"
_C.DATASET.TILE_OVERLAP = 0.3

# Stitching configuration.
_C.STITCHER.OUTPUT_PATH = "output/DEMIS/"
_C.STITCHER.RESOLUTION_RATIO = 0.5
_C.STITCHER.RESOLUTION_DIVISIBILITY = 8
_C.STITCHER.MAX_RESOLUTION = 2048
_C.STITCHER.MATCHING_METHOD = "loftr"  # loftr, sift or orb
_C.STITCHER.TRANSFORM_TYPE = "euclidean"  # translation, euclidean, similarity, affine or projective
_C.STITCHER.CONSTRUCTION_METHOD = "optimised"  # optimised, mst or slam
_C.STITCHER.COMPOSITING_METHOD = "overwrite"  # overwrite, average or adaptive
_C.STITCHER.OPTICAL_FLOW_REFINEMENT = True
_C.STITCHER.OPTICAL_FLOW_REFINEMENT_TYPE = "grid"  # mean or grid
_C.STITCHER.MAX_MATCHES = 1000
_C.STITCHER.MIN_MATCH_CONFIDENCE = 0.0
_C.STITCHER.MIN_MATCHES_AFTER_CONFIDENCE_TEST = 10
_C.STITCHER.ROOT_TILE = (0, 0)
_C.STITCHER.COLORED_OUTPUT = False
_C.STITCHER.SAVE_PAIRWISE_IMAGES = False
_C.STITCHER.SAVE_PAIRWISE_RESOLUTION_SCALE = 0.33
_C.STITCHER.SAVE_MATCHES = False
_C.STITCHER.SAVE_MATCHES_FRACTION = 1.0
_C.STITCHER.SAVE_MATCHES_COLOR_CODED = True
_C.STITCHER.EDGE_INFORMATION = (0.25, 0, 0, 0.25, 0, 3265) # Variance of 4 pixels for position and 1 degree for angle.
_C.STITCHER.RANSAC_THRESHOLD = 3.0
_C.STITCHER.RANSAC_TRIALS = 2000
_C.STITCHER.RANSAC_SEED = 42
_C.STITCHER.HESSIAN_THRESHOLD = 400
_C.STITCHER.DISTANCE_RATIO = 0.75
_C.STITCHER.FLANN_TREES = 5
_C.STITCHER.FLANN_CHECKS = 50
_C.STITCHER.NORMALISE_INTENSITY = True
_C.STITCHER.CLAHE_LIMIT = 0.5
_C.STITCHER.CLAHE_GRID = (4, 4)

# LoFTR configuration.
_C.LOFTR.CHECKPOINT_PATH = "LoFTR/weights/demis_ds.ckpt"

# Evaluator configuration.
_C.EVAL.SPLIT_PATH = "datasets/DEMIS/splits/test_list.txt"
_C.EVAL.ERROR_THRESHOLDS = (3, 5, 10)
_C.EVAL.INVERT_THRESHOLD = False
_C.EVAL.RAFT_RESOLUTION_RATIO = 0.5


cfg = _C


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for the DEMIS pipeline."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
