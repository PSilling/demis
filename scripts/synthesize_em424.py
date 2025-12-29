"""Synthesizes the EM424 dataset.

Project: Deep Electron Microscopy Image Stitching (DEMIS)
Author: Petr Šilling
Year: 2023
"""

import argparse
import re

from src.dataset.em424_synthesizer import EM424Synthesizer, EM424SynthesizerConfig

if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser(description="EM424 dataset synthesizer")
    parser.add_argument("src_path", type=str, help="path to the directory with source images")
    parser.add_argument("out_path", type=str, help="output directory path")
    parser.add_argument(
        "-o",
        "--base-overlap",
        type=float,
        default=0.2,
        help="base overlap between generated images",
    )
    parser.add_argument(
        "-r",
        "--tile-resolution",
        type=str,
        default="1024x1024",
        help="resolution (WxH) of generated images",
    )
    parser.add_argument(
        "-t",
        "--translate-contribution",
        type=float,
        default=0.03,
        help="maximum random translate contribution based on tile size",
    )
    parser.add_argument(
        "-a",
        "--rotation-angle",
        type=int,
        default=5,
        help="maximum rotation angle in degrees",
    )
    parser.add_argument(
        "-c",
        "--contrast-variance",
        type=float,
        default=0.0033,
        help="variance for random contrast changes",
    )
    parser.add_argument(
        "-b",
        "--brightness-variance",
        type=int,
        default=75,
        help="variance for random brightness changes",
    )
    parser.add_argument(
        "-n",
        "--noise-variance",
        type=int,
        default=25,
        help="variance for Gaussian noise",
    )
    args = parser.parse_args()

    # Prepare synthesizer configuration.
    config = EM424SynthesizerConfig()
    config.INPUT_PATH = args.src_path
    config.OUTPUT_PATH = args.out_path
    config.OVERLAP = min(max(args.base_overlap, 0.05), 0.95)
    config.AUGMENTATIONS = {
        "translate": min(max(args.translate_contribution, 0.0), 1.0),
        "rotate": min(max(args.rotation_angle, 0), 90),
        "contrast": max(args.contrast_variance, 0.0),
        "brightness": max(args.brightness_variance, 0),
        "gaussian_noise": max(args.noise_variance, 0),
    }

    # Parse the expected tile resolution.
    match = re.match(r"(\d+)x(\d+)", args.tile_resolution, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Invalid resolution specification: {args.tile_resolution}")
    config.TILE_RESOLUTION = tuple(int(x) for x in match.groups())

    synthesizer = EM424Synthesizer(config)
    synthesizer.synthesize_em424()
