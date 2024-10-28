"""Evaluates the DEMIS tool on the DEMIS dataset.

Project: Deep Electron Microscopy Image Stitching (DEMIS)
Author: Petr Šilling
Year: 2023
"""

import argparse
import os

from src.config.config import get_cfg_defaults
from src.eval.demis_evaluator import DEMISEvaluator

if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser(description="DEMIS tool evaluator")
    parser.add_argument("cfg_path", type=str, help="path to evaluation configuration")
    parser.add_argument(
        "-m",
        "--eval-matching",
        action="store_true",
        help="evaluate pairwise feature matching accuracy",
    )
    parser.add_argument(
        "-t",
        "--eval-homography",
        action="store_true",
        help="evaluate pairwise homography transformation accuracy",
    )
    parser.add_argument(
        "-p",
        "--eval-pairs",
        action="store_true",
        help="evaluate pairwise stitching output",
    )
    parser.add_argument(
        "-g",
        "--eval-grid",
        action="store_true",
        help="evaluate complete grid stitching output",
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

    # Parse the configuration file.
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_path)

    # If no method is selected, evaluate everything.
    eval_matching = args.eval_matching
    eval_homography = args.eval_homography
    eval_pairs = args.eval_pairs
    eval_grid = args.eval_grid
    if not eval_matching and not eval_homography and not eval_pairs and not eval_grid:
        eval_matching = True
        eval_homography = True
        eval_pairs = True
        eval_grid = True

    # Run the evaluation.
    evaluator = DEMISEvaluator(cfg, eval_matching, eval_homography, eval_pairs, eval_grid, args.count)
    evaluator.evaluate()
