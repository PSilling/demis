# Configuration Guide

DEMIS uses [YACS](https://github.com/rbgirshick/yacs) configuration files (in `configs/`) to control the stitching process, dataset handling, and evaluation. This document describes the most important configuration options.

## Structure

The configuration options are divided into four main sections:

- **DATASET**: Input dataset settings.
- **STITCHER**: Stitching parameters.
- **LOFTR**: [LoFTR](https://github.com/zju3dv/LoFTR) model settings.
- **EVAL**: Evaluation settings.

## DATASET Configuration

| Parameter      | Type    | Default             | Description                                                                     |
| -------------- | ------- | ------------------- | ------------------------------------------------------------------------------- |
| `PATH`         | String  | `"datasets/EM424/"` | Path to the input dataset directory.                                            |
| `TILE_OVERLAP` | Float   | `0.3`               | Expected overlap between adjacent tiles.                                        |
| `ROWS`         | Integer | `2`                 | Number of rows in the grid. Can be inferred from the dataset directory name.    |
| `COLS`         | Integer | `2`                 | Number of columns in the grid. Can be inferred from the dataset directory name. |

## STITCHER Configuration

### Image Matching

| Parameter          | Type    | Default       | Description                                                                                                                                                             |
| ------------------ | ------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RESOLUTION_RATIO` | Float   | `0.5`         | Resolution downscaling factor for feature matching.                                                                                                                     |
| `MAX_RESOLUTION`   | Integer | `2048`        | Maximum resolution per dimension for image processing.                                                                                                                  |
| `MATCHING_METHOD`  | String  | `"loftr"`     | Feature matching method: `"loftr"`, `"sift"`, or `"orb"`.                                                                                                               |
| `TRANSFORM_TYPE`   | String  | `"euclidean"` | Type of estimated geometric transformation: `"translation"`, `"euclidean"`, `"similarity"`, `"affine"`, or `"projective"`. Not all are valid for all stitching options. |

### Grid Construction and Compositing

| Parameter             | Type   | Default           | Description                                                                                                                             |
| --------------------- | ------ | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `CONSTRUCTION_METHOD` | String | `"optimised"`     | Grid construction method: `"optimised"` (global least squares optimisation), `"mst"` (Minimum Spanning Tree), or `"slam"` (SLAM graph). |
| `COMPOSITING_METHOD`  | String | `"overwrite"`     | Image blending method: `"overwrite"`, `"average"`, or `"adaptive"` (adaptive weights based on distance from seam).                      |
| `OUTPUT_PATH`         | String | `"output/EM424/"` | Output directory for stitching results.                                                                                                 |

### Optical Flow Refinement

| Parameter                      | Type    | Default  | Description                                                                                                           |
| ------------------------------ | ------- | -------- | --------------------------------------------------------------------------------------------------------------------- |
| `OPTICAL_FLOW_REFINEMENT`      | Boolean | `True`   | Enable refinement of transformations using optical flow ([RAFT](https://github.com/princeton-vl/RAFT) model).         |
| `OPTICAL_FLOW_REFINEMENT_TYPE` | String  | `"grid"` | Flow refinement type: `"grid"` (sample new points from the overlap region) or `"mean"` (use mean optical flow value). |

### Pre-processing and Debug

| Parameter              | Type    | Default | Description                                           |
| ---------------------- | ------- | ------- | ----------------------------------------------------- |
| `NORMALISE_INTENSITY`  | Boolean | `True`  | Apply intensity normalization to tiles.               |
| `COLORED_OUTPUT`       | Boolean | `False` | Colorize tiles in the final stitch for visualization. |
| `SAVE_MATCHES`         | Boolean | `False` | Save visualizations of feature matches.               |
| `SAVE_PAIRWISE_IMAGES` | Boolean | `False` | Save intermediate pairwise stitched images.           |

## LOFTR Configuration

| Parameter         | Type   | Default                         | Description                      |
| ----------------- | ------ | ------------------------------- | -------------------------------- |
| `CHECKPOINT_PATH` | String | `"LoFTR/weights/demis_ds.ckpt"` | Path to the LoFTR model weights. |

## EVAL Configuration

| Parameter               | Type   | Default                                 | Description                                 |
| ----------------------- | ------ | --------------------------------------- | ------------------------------------------- |
| `SPLIT_PATH`            | String | `"datasets/EM424/splits/test_list.txt"` | Path to EM424 test split file.              |
| `ERROR_THRESHOLDS`      | List   | `[3, 5, 10]`                            | AUC error thresholds in pixels.             |
| `RAFT_RESOLUTION_RATIO` | Float  | `0.5`                                   | Resolution ratio for RAFT-based evaluation. |
