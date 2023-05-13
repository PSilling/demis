"""Stitches grids of EM images using the DEMIS tool.

Project: Deep Electron Microscopy Image Stitching (DEMIS)
Author: Petr Šilling
Year: 2023
"""
import argparse
import cv2
import os
import re
from src.config.config import get_cfg_defaults
from src.dataset.dataset_loader import DatasetLoader
from src.dataset.demis_loader import DemisLoader
from src.pipeline.demis_stitcher import DemisStitcher
from src.pipeline.image_loader import ImageLoader


if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser(
        description="DEMIS tool for stitching grids of electron microscopy images"
    )
    parser.add_argument(
        "cfg_path", type=str, help="path to DEMIS stitching configuration"
    )
    parser.add_argument(
        "-d",
        "--use-demis-labels",
        action="store_true",
        help="stitch the DEMIS dataset using its labels",
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
    args = parser.parse_args()

    # Check path validity.
    if not os.access(args.cfg_path, os.R_OK):
        raise ValueError(f"Cannot read configuration file {args.cfg_path}.")

    # Prepare the stitching configuration.
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_path)
    cfg.freeze()

    # Check is the DEMIS dataset is in use.
    images_path = os.path.join(cfg.DATASET.PATH, "images")
    labels_path = os.path.join(cfg.DATASET.PATH, "labels")
    is_demis = os.path.isdir(images_path) and os.path.isdir(labels_path)

    # Load image paths.
    if is_demis:
        loader = DemisLoader(cfg.DATASET.PATH)
        labels = loader.load_labels()
        image_paths = loader.load_paths(labels)
    else:
        loader = DatasetLoader(cfg.DATASET.PATH)
        image_paths = loader.load_paths()

    # Parse the selected grid and slice indices.
    selected_grids = [int(i) for i in args.grid_indices] if args.grid_indices else None
    selected_slices = (
        [int(i) for i in args.slice_indices] if args.slice_indices else None
    )

    # Setup the output directory.
    os.makedirs(cfg.STITCHER.OUTPUT_PATH, exist_ok=True)

    # Stitch image tiles in the selected grids based on the given configuration.
    img_loader = ImageLoader(cfg)
    stitcher = DemisStitcher(cfg, img_loader)
    index = 1
    if is_demis and args.use_demis_labels:
        length = len(selected_grids) if selected_grids is not None else len(labels)
        for grid_labels in labels:
            # Select grids to stitch.
            match = re.search(r"g(\d+)", os.path.basename(grid_labels["path"]))
            if match is None:
                raise ValueError(
                    "Cannot parse labels file name: " f"{grid_labels['path']}."
                )
            grid_index = int(match.groups()[0])
            slice_index = 0  # The DEMIS dataset has no slices.
            if (
                selected_grids is not None and int(grid_index) not in selected_grids
            ) or (selected_slices is not None and slice_index not in selected_slices):
                continue

            # Report on current progress.
            print(
                f"[{index}/{length}] Stitching the grid starting with image "
                f"{grid_labels['tile_labels'][0]['path']}..."
            )
            index += 1

            # Stitch the grid and save the result.
            stitched_image, _ = stitcher.stitch_demis_grid_mst(grid_labels)
            out_filename = f"g{int(grid_index):05d}_s00000.png"
            out_path = os.path.join(cfg.STITCHER.OUTPUT_PATH, out_filename)
            cv2.imwrite(out_path, stitched_image)
    else:
        length = len(selected_grids) if selected_grids is not None else len(image_paths)
        for path_key, tile_paths in image_paths.items():
            # Select grids to stitch.
            grid_index, slice_index = path_key.split("_")
            if (
                selected_grids is not None and int(grid_index) not in selected_grids
            ) or (
                selected_slices is not None and int(slice_index) not in selected_slices
            ):
                continue

            # Report on current progress.
            print(
                f"[{index}/{length}] Stitching the grid starting with image "
                f"{tile_paths[0, 0]}..."
            )
            index += 1

            # Stitch the grid and save the result.
            stitched_image, _ = stitcher.stitch_grid(tile_paths)
            out_filename = f"g{int(grid_index):05d}_s{int(slice_index):05d}.png"
            out_path = os.path.join(cfg.STITCHER.OUTPUT_PATH, out_filename)
            cv2.imwrite(out_path, stitched_image)
