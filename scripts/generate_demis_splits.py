"""Synthesizes LoFTR splits and the corresponding LoFTR split indices
for the DEMIS dataset.

Project: Deep Electron Microscopy Image Stitching (DEMIS)
Author: Petr Šilling
Year: 2023
"""

import argparse
import os
import re

import numpy as np
from numpy.random import permutation

from src.config.config import get_cfg_defaults
from src.dataset.demis_loader import DemisLoader
from src.pipeline.demis_stitcher import DemisStitcher
from src.pipeline.image_loader import ImageLoader

if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser(description="DEMIS dataset splitter")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="path to DEMIS stitching configuration",
    )
    parser.add_argument(
        "-v",
        "--validation-grids",
        type=int,
        default=60,
        help="number of image grids for validation",
    )
    parser.add_argument(
        "-t",
        "--test-grids",
        type=int,
        default=60,
        help="number of image grids for testing",
    )
    args, extra_opts = parser.parse_known_args()

    # Prepare the evaluation configuration. Load the configuration file if given.
    cfg = get_cfg_defaults()
    if args.config and os.access(args.config, os.R_OK):
        cfg.merge_from_file(args.config)

    # Merge overrides specified via KEY=VALUE pairs.
    if extra_opts:
        opts = []
        for opt in extra_opts:
            opts.extend(opt.split("="))
        cfg.merge_from_list(opts)
    cfg.freeze()

    # Configure split sizes.
    train_split_size = None
    val_split_size = max(args.validation_grids, 0)
    test_split_size = max(args.test_grids, 0)

    # Configure paths.
    indices_dir = os.path.join(cfg.DATASET.PATH, "indices")
    splits_dir = os.path.join(cfg.DATASET.PATH, "splits")

    # Load the DEMIS labels.
    loader = DemisLoader(cfg.DATASET.PATH)
    labels = loader.load_labels()
    full_image_paths = loader.load_paths(labels)

    # Prepare for homography calculation.
    cache = ImageLoader(cfg)
    stitcher = DemisStitcher(cfg, cache)

    # Generate a set of LoFTR indices for each labels file.
    split_filenames = []
    os.makedirs(indices_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)
    for i, grid_labels in enumerate(labels):
        match = re.search(r"g(\d+)", os.path.basename(grid_labels["path"]))
        if match is None:
            raise ValueError("Cannot parse labels file name: " f"{grid_labels['path']}.")
        grid_index = int(match.groups()[0])
        grid_title = os.path.splitext(os.path.basename(grid_labels["path"]))[0]
        split_filenames.append(grid_title)
        tile_labels = grid_labels["tile_labels"]
        tile_paths = full_image_paths[f"{grid_index}_0"]
        print(f"[{i + 1}/{len(labels)}] Generating indices for labels file " f"{grid_title}...")

        # Go over all possible image pairs in the grid (scene) and add them to the
        # scene info.
        image_paths = []
        pair_infos = []
        poses = []
        for (row, col), tile_path in np.ndenumerate(tile_paths):
            # Save filepath.
            image_paths.append(os.path.basename(tile_path))

            # Add index info.
            grid_size = tile_paths.shape
            index = row * grid_size[1] + col
            if row < (grid_size[0] - 1):
                pair_infos.append(np.array([index, (row + 1) * grid_size[1] + col]))
            if col < (grid_size[1] - 1):
                pair_infos.append(np.array([index, index + 1]))

            # Add the pose based on the ground truth homography (depth is assumed
            # constant at 1).
            pose = np.identity(4)
            homography_gt = stitcher.get_transformation_to_reference(
                tile_labels=tile_labels[index], grid_labels=grid_labels
            )
            pose[[0, 1, 3], :2] = homography_gt[:, :2]
            pose[[0, 1, 3], 3] = homography_gt[:, 2]
            poses.append(pose)

        # Save the indices.
        output_path = os.path.join(indices_dir, f"{grid_title}.npz")
        np.savez_compressed(
            output_path,
            image_paths=np.array(image_paths),
            pair_infos=np.array(pair_infos),
            poses=np.array(poses),
        )

    # Create split lists.
    if val_split_size + test_split_size >= len(labels):
        raise ValueError("Too many validation or test files.")
    train_split_size = len(labels) - val_split_size - test_split_size

    output_path_val = os.path.join(splits_dir, "val_list.txt")
    split_filenames_val = split_filenames[:val_split_size]
    with open(output_path_val, "w") as file_val:
        file_val.write("\n".join(permutation(split_filenames_val)))

    output_path_test = os.path.join(splits_dir, "test_list.txt")
    split_filenames_test = split_filenames[val_split_size : val_split_size + test_split_size]
    with open(output_path_test, "w") as file_test:
        file_test.write("\n".join(permutation(split_filenames_test)))

    output_path_train = os.path.join(splits_dir, "train_list.txt")
    split_filenames_train = split_filenames[val_split_size + test_split_size :]
    with open(output_path_train, "w") as file_train:
        file_train.write("\n".join(permutation(split_filenames_train)))
