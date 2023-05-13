import cv2
import math
import numpy as np
import os
import random
from glob import glob


class DEMISSynthesizerConfig:
    """DEMIS dataset synthesizer configuration."""

    INPUT_PATH = "datasets/DEMIS Source/"
    OUTPUT_PREFIX = "demis_"
    OUTPUT_PATH = "datasets/DEMIS/"
    INPUT_FILETYPE = "*"
    OUTPUT_FILETYPE = "tif"

    OVERLAP = 0.2  # Base overlap between generated images.
    TILE_RESOLUTION = (1024, 1024)  # Resolution (WxH) of generated image tiles.

    # Randomised data augmentations to use when generating images.
    AUGMENTATIONS = {
        "translate": 0.03,  # Maximum shift contribution based on tile size.
        "rotate": 5,  # Maximum rotation in degrees.
        "contrast": 0.0033,  # Contrast change variance.
        "brightness": 75,  # Brightness change variance.
        "gaussian_noise": 25,  # Gaussian variance.
    }


class DEMISSynthesizer:
    """Synthesizer for images in the DEMIS dataset."""

    def __init__(self, config):
        """DEMISSynthesizer constructor.

        :param config: DEMIS synthesizer configuration.
        """
        self.config = config

    def synthesize_demis(self):
        """Synthesize DEMIS all images."""
        # Prepare output directories.
        img_output_path = os.path.join(self.config.OUTPUT_PATH, "images")
        labels_output_path = os.path.join(self.config.OUTPUT_PATH, "labels")
        os.makedirs(img_output_path, exist_ok=True)
        os.makedirs(labels_output_path, exist_ok=True)

        # Synthesize the dataset.
        demis_paths = self._parse_demis_paths()
        for i, path in enumerate(demis_paths):
            print(
                f"[{i + 1}/{len(demis_paths)}] Processing source image "
                f"{os.path.basename(path)}..."
            )
            img = cv2.imread(path, cv2.IMREAD_COLOR)

            if img is None:
                print(f"Failed to read image: {path}")
                continue

            img_tiles, labels = self._split_image(img)

            # Save all generated image tiles.
            for j, img_tile in enumerate(img_tiles):
                filename = (
                    f"{self.config.OUTPUT_PREFIX}g{i:05d}_t{j:05d}_s00000"
                    f".{self.config.OUTPUT_FILETYPE}"
                )
                cv2.imwrite(os.path.join(img_output_path, filename), img_tile)

            # Save image labels.
            filename = f"{self.config.OUTPUT_PREFIX}g{i:05d}.txt"
            with open(os.path.join(labels_output_path, filename), "w") as labels_file:
                header = labels[0]
                labels_file.write(
                    f"{header[0]:<2d}\t{header[1]:<2d}\t"
                    f"{header[2]:<5d}\t{header[3]:<5d}\n"
                )
                for j, label in enumerate(labels[1:]):
                    tile_path = (
                        f"../images/{self.config.OUTPUT_PREFIX}g{i:05d}_"
                        f"t{j:05d}_s00000.{self.config.OUTPUT_FILETYPE}"
                    )
                    labels_file.write(
                        f"{tile_path}\t{label[0]:<2d}\t{label[1]:<2d}\t"
                        f"{label[2]:<5d}\t{label[3]:<5d}\t"
                        f"{label[4]:< 4d}\n"
                    )

    def _parse_demis_paths(self):
        """
        Parse paths to DEMIS images. A single image or a directory are supported.

        :return: List of paths to DEMIS images.
        """
        if os.path.isfile(self.config.INPUT_PATH):
            paths = [self.config.INPUT_PATH]
        elif os.path.isdir(self.config.INPUT_PATH):
            search_path = os.path.join(
                self.config.INPUT_PATH, f"**/*.{self.config.INPUT_FILETYPE}"
            )
            paths = glob(search_path, recursive=True)
        else:
            raise ValueError(f"Cannot read from path: {self.config.INPUT_PATH}")
        return paths

    def _random_rotation(self, img, center_position, previous_angle=None):
        """Randomly rotates the given image along the given rotation center.

        :param img: Image to rotate.
        :param center_position: Position of the rotation center.
        :param previous_angle: Previous rotation angle. If not present,
                               0 degree rotation will be selected.
        :return: Randomly rotated image and the selected rotation angle.
        """
        if previous_angle is None:
            return img, 0

        max_diff = self.config.AUGMENTATIONS["rotate"]
        angle_min = max(previous_angle - max_diff, -max_diff)
        angle_max = min(previous_angle + max_diff, max_diff)
        rotation_angle = random.randrange(angle_min, angle_max)
        M = cv2.getRotationMatrix2D(center_position, rotation_angle, 1)
        rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        return rotated_img, rotation_angle

    def _add_random_gaussian(self, img):
        """Adds random Gaussian noise to each pixel of the given image.

        :param img: Image to process.
        :return: Image with added noise.
        """
        deviation = self.config.AUGMENTATIONS["gaussian_noise"] ** 0.5
        noise = cv2.randn(np.empty_like(img), (0,) * 3, (deviation,) * 3)
        return cv2.add(img, noise)

    def _random_brightness_contrast(self, img):
        """Randomly changes brightness and contrast of the given image. Contrast
        gain is limited to range [0.5, 1.5] and brightness bias is limited to
        range [-50, 50].

        :param img: Image to process.
        :return: Image with added updated brightness and contrast.
        """
        img = img.astype(np.float32)
        if self.config.AUGMENTATIONS["contrast"] is not None:
            gain = 1 + random.gauss(0, self.config.AUGMENTATIONS["contrast"] ** 0.5)
            img *= np.clip(gain, 0.5, 1.5)

        if self.config.AUGMENTATIONS["brightness"] is not None:
            bias = random.gauss(0, self.config.AUGMENTATIONS["brightness"] ** 0.5)
            img += np.clip(bias, -50, 50)

        return np.clip(img, 0, 255).astype(np.uint8)

    def _split_image(self, img):
        """Splits image to a number of smaller tiles based on the given synthesizer
        configuration.

        :param img: Image to split.
        :return: List of generated image tiles and tiling labels.
        """
        # Get pixel shifts necessary to avoid black bars with random rotation.
        tile_resolution = self.config.TILE_RESOLUTION
        rotation_sin = np.sin(np.deg2rad(self.config.AUGMENTATIONS["rotate"]))
        rotation_shifts = (
            math.ceil(tile_resolution[1] * rotation_sin),
            math.ceil(tile_resolution[0] * rotation_sin),
        )

        # Calculate the grid size based on the given tile resolution.
        # Calculations are based on the following equation:
        # TILES * TILE_SIZE - (TILES - 1) * (OVERLAP - MAX_TRANSLATION
        # + ROTATION_SHIFT / TILE_SIZE) * TILE_SIZE
        # =
        # IMG_SIZE - 2 * ROTATION_SHIFT
        max_shifts = (
            (
                self.config.OVERLAP
                - self.config.AUGMENTATIONS["translate"]
                + rotation_shifts[0] / tile_resolution[0]
            ),
            (
                self.config.OVERLAP
                - self.config.AUGMENTATIONS["translate"]
                + rotation_shifts[1] / tile_resolution[1]
            ),
        )
        width_tiles = (img.shape[1] - 2 * rotation_shifts[0]) / tile_resolution[0]
        height_tiles = (img.shape[0] - 2 * rotation_shifts[1]) / tile_resolution[1]
        width_tiles -= max_shifts[0]
        height_tiles -= max_shifts[1]
        grid_size = (
            int(height_tiles / (1 - max_shifts[1])),
            int(width_tiles / (1 - max_shifts[0])),
        )

        # Calculate pixel shifts for overlap and random translation.
        overlap_shifts = (
            round(tile_resolution[0] * self.config.OVERLAP),
            round(tile_resolution[1] * self.config.OVERLAP),
        )
        translate_shifts = (
            math.ceil(tile_resolution[0] * self.config.AUGMENTATIONS["translate"]),
            math.ceil(tile_resolution[1] * self.config.AUGMENTATIONS["translate"]),
        )

        # Tile the input image.
        img_tiles = []
        previous_angle = None
        labels = [(grid_size[0], grid_size[1], tile_resolution[0], tile_resolution[1])]
        for i, j in np.ndindex(grid_size):
            # Get base start position without translation and rotation shifts.
            start_position = [
                j * (tile_resolution[0] - overlap_shifts[0]),
                i * (tile_resolution[1] - overlap_shifts[1]),
            ]

            # Add random rotation shift (plus one for image boundary and minus one
            # for each following pair of neighboring tiles).
            start_position[0] += rotation_shifts[0] * (1 - j)
            start_position[1] += rotation_shifts[1] * (1 - i)

            # Add random translation shift.
            if j > 0:
                start_position[0] += random.randrange(translate_shifts[0])
            if i > 0:
                start_position[1] += random.randrange(translate_shifts[1])

            # Find the center and end positions of the tile.
            center_position = (
                start_position[0] + tile_resolution[0] // 2,
                start_position[1] + tile_resolution[1] // 2,
            )
            end_position = (
                start_position[0] + tile_resolution[0],
                start_position[1] + tile_resolution[1],
            )

            # Apply random rotation around the tile center position.
            rotated_img, rotation_angle = self._random_rotation(
                img, center_position, previous_angle
            )
            previous_angle = rotation_angle

            # Crop the tile.
            img_tile = rotated_img[
                start_position[1] : end_position[1], start_position[0] : end_position[0]
            ].copy()

            # Apply all remaining data augmentations.
            if self.config.AUGMENTATIONS["gaussian_noise"] is not None:
                img_tile = self._add_random_gaussian(img_tile)
            img_tile = self._random_brightness_contrast(img_tile)

            # Save the tile and its start position label.
            img_tiles.append(img_tile)
            labels.append((i, j, start_position[0], start_position[1], rotation_angle))

        return img_tiles, labels
