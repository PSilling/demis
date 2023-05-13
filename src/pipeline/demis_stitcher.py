import cv2
import numpy as np
from src.pipeline.grid_stitcher import GridStitcher
from src.pipeline.tile_node import TileNode


class DemisStitcher(GridStitcher):
    """Specialised grid stitcher that provided additional methods to work
    with the DEMIS dataset."""

    def get_transformation_to_reference(self, tile_labels, grid_labels):
        """Get the transformation matrix from the coordinate space of one tile
        to the reference tile (first tile of the grid).

        :param tile_labels: Labels of the source tile.
        :param grid_labels: Labels of images in the grid that contains the tile labels.
        :return: Calculated transformation matrix.
        """
        # Use the initial tile as reference as it has no random rotation.
        reference_labels = grid_labels["tile_labels"][0]

        # Calculate translational shifts.
        x_shift = tile_labels["position"][0] - reference_labels["position"][0]
        y_shift = tile_labels["position"][1] - reference_labels["position"][1]
        T = np.array([[1, 0, x_shift], [0, 1, y_shift], [0, 0, 1]])

        # Get the rotation matrix.
        angle_difference = -tile_labels["angle"]
        rotation_center = (
            grid_labels["tile_resolution"][0] // 2 + x_shift,
            grid_labels["tile_resolution"][1] // 2 + y_shift,
        )
        R = np.identity(3)
        R[:2, :] = cv2.getRotationMatrix2D(rotation_center, angle_difference, 1)

        return R @ T

    def get_transformation_between_tiles(self, tile_labels1, tile_labels2, grid_labels):
        """Get the transformation matrix from the coordinate space of one tile
        to another.

        :param tile_labels1: Labels of the source tile.
        :param tile_labels2: Labels of the target tile.
        :param grid_labels: Labels of images in the grid that contains the tile labels.
        :return: Calculated transformation matrix.
        """
        # Get transformations to the reference tile.
        M1 = self.get_transformation_to_reference(tile_labels1, grid_labels)
        M2 = self.get_transformation_to_reference(tile_labels2, grid_labels)

        # Compute the transformation matrix to the target tile via the reference tile.
        M2_inv = np.linalg.inv(M2)
        return M2_inv @ M1

    def stitch_demis_grid_mst(self, grid_labels):
        """Stitch DEMIS images in a grid by constructing a minimum spanning tree (MST)
        based on the corresponding DEMIS labels.

        :param grid_labels: DEMIS labels of images in a single grid.
        :return: Stitched DEMIS grid image and the coordinates of the root tile.
        """
        # Get labels of the root tile.
        if self.cfg.STITCHER.ROOT_TILE:
            root_position = self.cfg.STITCHER.ROOT_TILE
        else:
            root_position = (
                grid_labels["grid_size"][0] // 2,
                grid_labels["grid_size"][1] // 2,
            )

        root_labels = [
            tile_labels
            for tile_labels in grid_labels["tile_labels"]
            if tile_labels["grid_position"] == root_position
        ]
        if not root_labels:
            raise ValueError(f"Invalid root tile position selected: {root_position}")
        root_labels = root_labels[0]

        # Create the root node.
        tile_nodes = [
            TileNode(
                cfg=self.cfg,
                img=self.img_loader.load_img(root_labels["path"]),
                position=root_position,
                transformation=np.identity(3),
            )
        ]

        # Connect all remaining tiles to the root node.
        for tile_labels in grid_labels["tile_labels"]:
            if tile_labels["grid_position"] != root_position:
                M = self.get_transformation_between_tiles(
                    tile_labels, root_labels, grid_labels
                )
                tile_nodes.append(
                    TileNode(
                        cfg=self.cfg,
                        img=self.img_loader.load_img(tile_labels["path"]),
                        position=tile_labels["grid_position"],
                        parent=tile_nodes[0],
                        transformation=M,
                    )
                )

        return self.stitch_mst_nodes(tile_nodes)
