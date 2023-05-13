"""Default DEMIS pipeline configuration."""

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
_C.STITCHER.MATCHING_METHOD = "loftr"
_C.STITCHER.TRANSFORM_TYPE = "affine"
_C.STITCHER.CONSTRUCTION_METHOD = "slam"
_C.STITCHER.COMPOSITING_METHOD = "overwrite"
_C.STITCHER.ROOT_TILE = (0, 0)
_C.STITCHER.COLORED_OUTPUT = False
_C.STITCHER.SAVE_MATCHES = False
_C.STITCHER.SAVE_MATCHES_FRACTION = 0.1
# Variance of 4 pixels for position and 1 degree for angle.
_C.STITCHER.EDGE_INFORMATION = (0.25, 0, 0, 0.25, 0, 3265)
_C.STITCHER.RANSAC_THRESHOLD = 3.0
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
_C.EVAL.MAX_MATCHES = 1000
_C.EVAL.ERROR_THRESHOLDS = (3, 5, 10)


cfg = _C


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for the DEMIS pipeline."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
