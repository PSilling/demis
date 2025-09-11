"""Tree node representing an image tile.

Project: Deep Electron Microscopy Image Stitching (DEMIS)
Author: Petr Šilling
Year: 2023
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import AffineTransform, EuclideanTransform, ProjectiveTransform, SimilarityTransform


class TileNode:
    """Tree node for an image tile."""

    def __init__(
        self,
        cfg,
        img,
        position,
        parent=None,
        matches=None,
        matches_parent=None,
        inliers_map=None,
        transformation=None,
    ):
        """TileNode constructor.

        :param cfg: DEMIS configuration.
        :param img: Image in the node.
        :param position: Position in the grid.
        :param parent: Parent node.
        :param matches: Matching keypoints for this node.
        :param matches_parent: Matching keypoints for the parent node.
        :param inliers_map: Boolean map specifying which matches are inliers.
        :param transformation: Transformation matrix for this node (calculated lazily).
        """
        self.cfg = cfg
        self.img = img
        self.position = position
        self.parent = parent
        self.matches = matches
        self.matches_parent = matches_parent
        self.inliers_map = inliers_map
        self.transformation = transformation
        if transformation is not None and parent is not None and parent.transformation is not None:
            self.transformation = parent.transformation.dot(transformation)

    def estimate_transformation(self):
        """Estimate image tile transformation based on the corresponding matching result
        and the parent transformation. Includes inlier selection."""
        # Estimate the transformation matrix of the selected transformation type.
        min_samples = 3
        if self.cfg.STITCHER.TRANSFORM_TYPE == "euclidean":
            transform_type = EuclideanTransform
        elif self.cfg.STITCHER.TRANSFORM_TYPE == "similarity":
            transform_type = SimilarityTransform
        elif self.cfg.STITCHER.TRANSFORM_TYPE == "affine":
            transform_type = AffineTransform
        elif self.cfg.STITCHER.TRANSFORM_TYPE == "projective":
            transform_type = ProjectiveTransform
            min_samples = 4
        else:
            raise ValueError(
                f"Transform type '{self.cfg.STITCHER.TRANSFORM_TYPE}' not supported for the selected method. "
                "Please choose one of: euclidean, similarity, affine, projective."
            )

        # Apply RANSAC to find the selected transformation and inlier keypoints.
        result, inliers_map = ransac(
            (self.matches, self.matches_parent),
            transform_type,
            min_samples=min_samples,
            residual_threshold=self.cfg.STITCHER.RANSAC_THRESHOLD,
            max_trials=self.cfg.STITCHER.RANSAC_TRIALS,
            rng=self.cfg.STITCHER.RANSAC_SEED,
        )

        M = result.params
        self.inliers_map = inliers_map

        # Add the parent transformation if present.
        if self.parent:
            M = self.parent.transformation.dot(M)
        self.transformation = M

    def get_pose(self):
        """Get the pose (position and angle) of the current transformation.

        :return: Updated translation values and angle.
        """
        if self.transformation is None:
            raise ValueError("Cannot calculate pose from an undefined transformation.")

        if self.cfg.STITCHER.TRANSFORM_TYPE not in ("translation", "euclidean"):
            raise ValueError("Poses can only be correctly calculated for translation or euclidean transformations.")

        M = self.transformation
        angle = np.arctan2(M[1, 0], M[1, 1])
        return M[0, 2], M[1, 2], angle

    def color_coat_image(self):
        """Add color coating to the image tile. Color coating is based on tile
        position."""
        # If already colored, skip.
        if self.img.ndim == 3:
            return
        
        # Define row color pallette.
        color_shifts = np.array(
            [
                [0.40, 0.30, 0.30],
                [0.30, 0.40, 0.30],
                [0.30, 0.30, 0.40],
                [0.45, 0.45, 0.10],
                [0.45, 0.10, 0.45],
                [0.10, 0.45, 0.45],
            ]
        )

        # Reverse the pallette for odd rows.
        if self.position[0] % 2 == 1:
            color_shifts = color_shifts[::-1]

        # Use column index to select a specific color target.
        color_shift = color_shifts[self.position[1] % color_shifts.shape[0]]

        # Add color coating by weighing colors by the selected color shift.
        self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB) * color_shift
        self.img = np.rint(self.img).astype("uint8")

    def get_warped_boundary(self):
        """Gets the bounding coordinates of the image tile after warping by the
        estimated transformation.

        :return: Bounding coordinates in the warped coordinate space: (minX, maxX), (minY, maxY).
        """
        # Get homogenous coordinates of the four image tile corners.
        if self.img.ndim == 2:
            height, width = self.img.shape
        else:
            height, width, _ = self.img.shape
        C = np.array([[0, width, width, 0], [0, 0, height, height], [1, 1, 1, 1]])

        # Warp the corner coordinates.
        C = self.transformation @ C

        # Convert the result back to 2D coordinates.
        C[0] /= C[2]
        C[1] /= C[2]
        C = C[:2]

        # Calculate bounding box boundaries.
        minXY = np.min(C, axis=1)
        maxXY = np.max(C, axis=1)
        return np.stack([minXY, maxXY]).T


@dataclass(order=True)
class WeightedTileNode:
    """Dataclass for tile nodes. Uses only node weights for prioritization."""

    weight: float
    node: TileNode = field(compare=False)
