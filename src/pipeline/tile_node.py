import cv2
import numpy as np
from dataclasses import dataclass, field


class TileNode:
    """Tree node for an image tile."""

    def __init__(self, cfg, img, position, parent=None, matches=None,
                 matches_parent=None, transformation=None):
        """TileNode constructor.

        :param cfg: DEMIS configuration.
        :param img: Image in the node.
        :param position: Position in the grid.
        :param parent: Parent node.
        :param matches: Matching keypoints for this node.
        :param matches_parent: Matching keypoints for the parent node.
        :param transformation: Transformation matrix for this node (calculated lazily).
        """
        self.cfg = cfg
        self.img = img
        self.position = position
        self.parent = parent
        self.matches = matches
        self.matches_parent = matches_parent
        self.transformation = transformation

    def estimate_transformation(self):
        """Estimate image tile transformation based on the corresponding matching result
        and the parent transformation."""
        # Estimate the transformation matrix of the selected transformation type.
        # The resulting matrix is always converted to a complete 3x3 shape.
        if self.cfg.STITCHER.TRANSFORM_TYPE == "affine":
            M = np.identity(3)
            M[:2, :], _ = cv2.estimateAffine2D(
                self.matches,
                self.matches_parent,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.cfg.STITCHER.RANSAC_THRESHOLD
            )
        elif self.cfg.STITCHER.TRANSFORM_TYPE == "perspective":
            M, _ = cv2.findHomography(self.matches, self.matches_parent, cv2.RANSAC,
                                      self.cfg.STITCHER.RANSAC_THRESHOLD)
        else:
            raise ValueError("Invalid transform type: "
                             f"{self.cfg.STITCHER.TRANSFORM_TYPE}")

        # Add the parent transformation if present.
        if self.parent:
            M = self.parent.transformation.dot(M)
        self.transformation = M

    def color_coat_image(self):
        """Add color coating to the image tile. Color coating is based on tile
        position."""
        # Define row color pallette.
        color_shifts = np.array([[0.40, 0.30, 0.30],
                                 [0.30, 0.40, 0.30],
                                 [0.30, 0.30, 0.40],
                                 [0.45, 0.45, 0.10],
                                 [0.45, 0.10, 0.45],
                                 [0.10, 0.45, 0.45]])

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

        :return: Bounding coordinates in the warped coordinate space:
                 (minX, maxX), (minY, maxY).
        """
        # Get homogenous coordinates of the four image tile corners.
        height, width = self.img.shape
        C = np.array([[0, width,  width,      0],
                      [0,     0, height, height],
                      [1,     1,      1,      1]])

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
