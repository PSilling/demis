"""Stitches grids of EM images using the DEMIS tool.

Project: Deep Electron Microscopy Image Stitching (DEMIS)
Author: Petr Šilling
Year: 2023
"""

import argparse
import os
import re
import warnings

import cv2

from src.config.config import get_cfg_defaults
from src.dataset.dataset_loader import DatasetLoader
from src.dataset.em424_loader import EM424Loader
from src.pipeline.em424_stitcher import EM424Stitcher
from src.pipeline.image_loader import ImageLoader

if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser(description="DEMIS tool for stitching grids of electron microscopy images")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="path to DEMIS stitching configuration",
    )
    parser.add_argument(
        "-d",
        "--use-em424-labels",
        action="store_true",
        help="stitch the EM424 dataset using its labels",
    )
    parser.add_argument(
        "-g",
        "--grid-index",
        dest="grid_indices",
        action="append",
        help="index of a grid to stitch (can be used multiple times)",
    )
    parser.add_argument(
        "-s",
        "--slice-index",
        dest="slice_indices",
        action="append",
        help="index of a slice to stitch (can be used multiple times)",
    )
    parser.add_argument(
        "-p",
        "--plugin-mode",
        action="store_true",
        help="run in plugin mode (for ImageJ integration)",
    )

    args, extra_opts = parser.parse_known_args()

    # Prepare the stitching configuration. Load the configuration file if given.
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

    # Suppress all warnings if in plugin mode.
    if args.plugin_mode:
        warnings.filterwarnings("ignore")

    # Check if the EM424 dataset is in use.
    images_path = os.path.join(cfg.DATASET.PATH, "images")
    labels_path = os.path.join(cfg.DATASET.PATH, "labels")
    is_em424 = os.path.isdir(images_path) and os.path.isdir(labels_path) and not args.plugin_mode

    # Load image paths.
    if is_em424:
        loader = EM424Loader(cfg.DATASET.PATH)
        labels = loader.load_labels()
        image_paths = loader.load_paths(labels)
    else:
        loader = DatasetLoader(cfg.DATASET.PATH, cfg.DATASET.ROWS, cfg.DATASET.COLS)
        image_paths = loader.load_paths()

    # Parse the selected grid and slice indices.
    selected_grids = [int(i) for i in args.grid_indices] if args.grid_indices else None
    selected_slices = [int(i) for i in args.slice_indices] if args.slice_indices else None

    # Setup the output directory.
    os.makedirs(cfg.STITCHER.OUTPUT_PATH, exist_ok=True)

    # Stitch image tiles in the selected grids based on the given configuration.
    img_loader = ImageLoader(cfg)
    stitcher = EM424Stitcher(cfg, img_loader)
    index = 1
    if is_em424 and args.use_em424_labels:
        length = len(selected_grids) if selected_grids is not None else len(labels)
        for grid_labels in labels:
            # Select grids to stitch.
            match = re.search(r"g(\d+)", os.path.basename(grid_labels["path"]))
            if match is None:
                raise ValueError(f"Cannot parse labels file name: {grid_labels['path']}.")
            grid_index = int(match.groups()[0])
            slice_index = 0  # The EM424 dataset has no slices.
            if (selected_grids is not None and int(grid_index) not in selected_grids) or (
                selected_slices is not None and slice_index not in selected_slices
            ):
                continue

            # Report on current progress.
            print(
                f"[{index}/{length}] Stitching the grid starting with image {grid_labels['tile_labels'][0]['path']}..."
            )
            index += 1

            # Stitch the grid and save the result.
            stitched_image, _ = stitcher.stitch_em424_grid_mst(grid_labels)
            out_filename = f"g{int(grid_index):05d}_s00000.png"
            out_path = os.path.join(cfg.STITCHER.OUTPUT_PATH, out_filename)
            cv2.imwrite(out_path, stitched_image)
    else:
        length = len(selected_grids) if selected_grids is not None else len(image_paths)
        for path_key, tile_paths in image_paths.items():
            # Select grids to stitch.
            grid_index, slice_index = path_key.split("_")
            if (selected_grids is not None and int(grid_index) not in selected_grids) or (
                selected_slices is not None and int(slice_index) not in selected_slices
            ):
                continue

            # Report on current progress.
            print(f"[{index}/{length}] Stitching the grid starting with image {tile_paths[0, 0]}...")
            index += 1

            # Stitch the grid and save the result.
            stitched_image, _ = stitcher.stitch_grid(tile_paths)
            out_filename = f"g{int(grid_index):05d}_s{int(slice_index):05d}.png"
            out_path = os.path.join(cfg.STITCHER.OUTPUT_PATH, out_filename)
            cv2.imwrite(out_path, stitched_image)
