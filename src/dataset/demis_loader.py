import re
import os
from glob import glob
from src.dataset.dataset_loader import DatasetLoader


class DemisLoader(DatasetLoader):
    """Data loader for images in the DEMIS dataset."""

    def parse_demis_labels(self, path):
        """Parses the given DEMIS dataset labels file.

        :param path: Path to the labels file.
        :return: Parsed label data.
        """
        with open(path) as labels_file:
            lines = labels_file.readlines()

            # Parse metadata on the first line. The metadata are structured as follows:
            # <rows> <columns> <tile_width> <tile_height>
            number_regex = r"(-?\d+)\s+"
            match = re.match(number_regex * 4, lines[0])
            if match is None:
                raise ValueError(f"Invalid metadata in labels file: {path}.")
            grid_size = tuple(int(x) for x in match.groups()[:2])
            tile_resolution = tuple(int(x) for x in match.groups()[2:])

            # Parse image tile labels. Labels are structured as follows:
            # <path> <row> <column> <x_axis_position> <y_axis_position> <rotation_angle>
            tile_labels = []
            for line in lines[1:]:
                match = re.match(r"(.*)\s+" + number_regex * 5, line)
                if match is None:
                    raise ValueError(f"Invalid image tile labels in file: {path}.")
                groups = match.groups()
                tile_labels.append({
                    "path": os.path.normpath(
                        os.path.join(os.path.dirname(path), groups[0])
                    ),
                    "grid_position": (int(groups[1]), int(groups[2])),
                    "position": (int(groups[3]), int(groups[4])),
                    "angle": int(groups[5])
                })

            # Order the tile labels based on their grid positions.
            tile_labels = sorted(tile_labels,
                                 key=lambda l: f"{l['grid_position'][0]}"
                                               f"{l['grid_position'][1]}")

            return {
                "path": path,
                "grid_size": grid_size,
                "tile_resolution": tile_resolution,
                "tile_labels": tile_labels
            }

    def load_labels(self, split_path=None):
        """
        Loads labels of all DEMIS images.

        :param split_path: Path to a split file. If included, only labels from the
                           split are loaded.
        :return: Parsed DEMIS labels.
        """
        # Check if the DEMIS directory exists and has the correct structure.
        labels_path = os.path.join(self.path, "labels")
        images_path = os.path.join(self.path, "images")
        if not os.path.isdir(self.path):
            raise ValueError(f"Cannot read directory: {self.path}")
        if not os.path.isdir(labels_path) or not os.path.isdir(images_path):
            raise ValueError("The given DEMIS directory has an unexpected structure.")

        # Get paths to all labels files.
        labels_paths = glob(os.path.join(labels_path, "*.txt"))

        # Limit label paths to those in the split if one was given.
        if split_path:
            with open(split_path) as split_file:
                splits = split_file.read().splitlines()
            splits = [os.path.join(labels_path, path + ".txt") for path in splits]
            labels_paths = [path for path in labels_paths if path in splits]

        labels = [self.parse_demis_labels(path) for path in labels_paths]
        return labels

    def load_paths(self, labels=None):
        """
        Loads paths to all DEMIS images based on the corresponding label files.

        :param labels: Optional list of preloaded DEMIS labels.
        :return: Dictionary of 2D arrays of image paths for each grid.
        """
        # Get DEMIS labels.
        if not labels:
            labels = self.load_labels()

        # Convert labels to a image path dictionary.
        image_paths = {}
        for grid_labels in labels:
            paths = [tile_labels["path"] for tile_labels in grid_labels["tile_labels"]]
            image_paths.update(self.get_image_paths(paths, grid_labels["grid_size"]))
        return image_paths
