"""Converts results of other stitching to a format that DEMIS can parse.
Result files need to contain grid and slice numbers in their filenames.

Currently supported stitching tools:
- MIST (https://github.com/usnistgov/MIST)

Project: Deep Electron Microscopy Image Stitching (DEMIS)
Author: Petr Šilling
Year: 2024
"""

import argparse
import json
import os
import re
from glob import glob


def convert_mist(input_path: str, output_path: str):
    """Converts results from MIST to DEMIS format.

    :param input_path: Input directory path.
    :param output_path: Output directory path.
    """
    print("Converting MIST results...")

    os.makedirs(output_path, exist_ok=True)
    for results_path in glob(os.path.join(input_path, "*global-positions-*.txt")):
        # Parse grid and slice numbers from the filename.
        match_grid = re.search(r"g(\d+)", results_path, flags=re.IGNORECASE)
        match_slice = re.search(r"s(\d+)", results_path, flags=re.IGNORECASE)
        if match_grid is None or match_slice is None:
            raise ValueError(f"Results file {results_path} does not contain grid and slice info.")
        grid_index = int(match_grid.group()[1:])
        slice_index = int(match_slice.group()[1:])

        # Parse tile positions from the results file and convert them to translation matrices. MIST results format:
        # file: <filename>; corr: <corr>; position: <position_x, position_y>; grid: <row, column>;
        global_transformations = []
        with open(results_path) as results_file:
            lines = results_file.readlines()
            for line in lines:
                match = re.findall(r"\(-?(\d+), (-?\d+)\)", line)
                if not match:
                    raise ValueError(f"Invalid position data in results file: {results_path}.")
                global_position = (int(match[0][0]), int(match[0][1]))
                grid_position = (int(match[1][1]), int(match[1][0]))

                # Convert the global position a translation matrix.
                T = [[1, 0, global_position[0]], [0, 1, global_position[1]], [0, 0, 1]]
                global_transformations.append({"position": grid_position, "transformation": T})

        # Output the converted data in JSON format.
        with open(os.path.join(output_path, f"g{grid_index:05d}_s{slice_index:05d}.json"), "w") as output_file:
            json.dump(global_transformations, output_file, indent=4)

    print("Conversion finished.")


if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser(description="DEMIS tool results converter")
    parser.add_argument("type", type=str, help="type of results format (supported values: mist)")
    parser.add_argument("input_path", type=str, help="path to the input directory")
    parser.add_argument("output_path", type=str, help="path to the output directory")
    args = parser.parse_args()

    type = args.type.lower()
    if type not in ["mist"]:
        raise ValueError("Results type must be one of ['mist'].")

    # Check parameter validity.
    if not os.path.isdir(args.input_path):
        raise ValueError(f"Cannot access the input directory: {args.input_path}.")

    # Run the conversion.
    if type:
        convert_mist(args.input_path, args.output_path)
