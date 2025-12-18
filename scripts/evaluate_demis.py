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
    parser = argparse.ArgumentParser(description="DEMIS tool evaluator for the DEMIS dataset")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="path to evaluation configuration",
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=None,
        help="maximum number of images to evaluate",
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

    # Run the evaluation.
    evaluator = DEMISEvaluator(cfg, args.number)
    evaluator.run()
