"""Generic dataset loader.

Project: Deep Electron Microscopy Image Stitching (DEMIS)
Author: Petr Šilling
Year: 2023
"""

import os
import re
from glob import glob

import numpy as np


class DatasetLoader:
    """Standard class for loading datasets of EM image tiles from a directory."""

    def __init__(self, path, rows=None, cols=None):
        """DatasetLoader constructor.

        :param path: Path to the base dataset directory.
        :param rows: Number of rows in the grid. Inferred from directory name if None.
        :param cols: Number of columns in the grid. Inferred from directory name if None.
        """
        self.path = path
        self.rows = rows
        self.cols = cols

    def parse_grid_size(self):
        """Parses the expected grid size from the current dataset directory path.

        :return: Parsed grid size (rows, columns).
        """
        dir_name = os.path.basename(os.path.normpath(self.path))
        match = re.match(r"(\d+)x(\d+)", dir_name, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"Invalid grid specification: {dir_name}'")
        return tuple(int(x) for x in match.groups())

    def parse_grid_info(self, tile_path):
        """Parses grid information from the path to an image tile.

        :param tile_path: Image tile path.
        :return: Parsed grid information (grid_index, tile_index, slice_index).
        """
        match = re.match(r".*g(\d+).t(\d+).s(\d+).*", tile_path, flags=re.IGNORECASE)
        if match is None:
            return None
        return tuple(int(x) for x in match.groups())

    def get_image_paths(self, paths, grid_size):
        """Builds arrays of image paths separated by grid membership. Images have to
        share the same grid specification.

        :param paths: Paths to analyse.
        :param grid_size: Expected grid size.
        :return: Dictionary of 2D arrays of image paths for each grid.
        """
        image_paths = {}
        for path in paths:
            # Read grid, tile and slice info.
            grid_info = self.parse_grid_info(path)
            if grid_info is None:
                continue
            grid_index, tile_index, slice_index = grid_info

            # Create the 2D paths array if not already present. The arrays represent
            # sets of image tiles and are separated by grid and slice indices.
            path_key = f"{grid_index}_{slice_index}"
            if path_key not in image_paths:
                image_paths[path_key] = np.zeros(grid_size, dtype=object)

            # Register the path to its appropriate position.
            row = tile_index // grid_size[1]
            column = tile_index % grid_size[1]
            if row < grid_size[0]:
                image_paths[path_key][row, column] = path

        # Validate that the number of paths corresponds to the expected grid size.
        for path_key in image_paths:
            if 0 in image_paths[path_key]:
                grid_index, slice_index = path_key.split("_")
                raise ValueError(
                    f"Images at grid index g{int(grid_index):05d} "
                    f"and slice index s{int(slice_index):05d} do "
                    "not match the expected grid size "
                    f"{grid_size[0]}x{grid_size[1]}."
                )
        return image_paths

    def load_paths(self):
        """
        Loads paths to images in the given dataset directory.

        :return: Dictionary of 2D arrays of image paths for each grid.
        """
        # Check if the target directory exists.
        if not os.path.isdir(self.path):
            raise ValueError(f"Cannot read directory: {self.path}")

        grid_size = (self.rows, self.cols) if self.rows and self.cols else self.parse_grid_size()
        paths = glob(os.path.join(self.path, "*.*"))
        return self.get_image_paths(paths, grid_size)
