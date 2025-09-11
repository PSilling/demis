"""Evaluates the DEMIS tool on the a given grid of EM images.

Project: Deep Electron Microscopy Image Stitching (DEMIS)
Author: Petr Šilling
Year: 2024
"""

import argparse
import os

from src.config.config import get_cfg_defaults
from src.eval.grid_evaluator import GridEvaluator

if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser(description="DEMIS tool evaluator for generic datasets")
    parser.add_argument("cfg_path", type=str, help="path to evaluation configuration")
    parser.add_argument(
        "-t",
        "--transformations-path",
        type=str,
        default="",
        help="path to a directory with precomputed global tile transformations",
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=None,
        help="maximum number of images to evaluate",
    )
    args = parser.parse_args()

    # Check path validity.
    if not os.access(args.cfg_path, os.R_OK):
        raise ValueError(f"Cannot read configuration file {args.cfg_path}.")
    
    if args.transformations_path and not os.path.isdir(args.transformations_path):
        raise ValueError(f"Cannot access transformations directory {args.transformations_path}.")

    # Parse the configuration file.
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_path)

    # Run the evaluation.
    evaluator = GridEvaluator(cfg, args.transformations_path, args.count)
    evaluator.run()
