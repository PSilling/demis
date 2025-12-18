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
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="path to evaluation configuration",
    )
    parser.add_argument(
        "-t",
        "--transformations-path",
        type=str,
        default="",
        help="path to a directory with precomputed global tile transformations",
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=None,
        help="maximum number of images to evaluate",
    )
    args, extra_opts = parser.parse_known_args()

    # Check path validity.
    if args.transformations_path and not os.path.isdir(args.transformations_path):
        raise ValueError(f"Cannot access transformations directory {args.transformations_path}.")

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

    # Run the evaluation.
    evaluator = GridEvaluator(cfg, args.transformations_path, args.number)
    evaluator.run()
