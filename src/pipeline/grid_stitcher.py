import cv2
import numpy as np
import torch
from queue import PriorityQueue, Empty
from LoFTR.src.loftr import LoFTR, default_cfg
from src.pipeline.tile_node import TileNode, WeightedTileNode
from src.pipeline.tile_stitcher import TileStitcher


class GridStitcher():
    """Stitcher for grids of overlapping images."""

    def __init__(self, cfg, cache):
        """GridStitcher constructor.

        :param cfg: DEMIS configuration.
        :param cache: Image cache.
        """
        self.cfg = cfg
        self.cache = cache
        self.img_processors = {"loftr": None,
                               "sift": None,
                               "flann": None}

        # Setup the matcher instance.
        if cfg.STITCHER.MATCHING_METHOD == "loftr":
            # Prepare LoFTR for matching using outdoor weights (better than indoor).
            self.img_processors["loftr"] = LoFTR(config=default_cfg)
            self.img_processors["loftr"].load_state_dict(
                torch.load(cfg.LOFTR.CHECKPOINT_PATH)["state_dict"]
            )
            self.img_processors["loftr"] = self.img_processors["loftr"].eval().cuda()
        elif cfg.STITCHER.MATCHING_METHOD == "sift":
            self.img_processors["sift"] = cv2.SIFT_create()
            index_params = {"algorithm": 1, "trees": cfg.STITCHER.FLANN_TREES}
            search_params = {"checks": cfg.STITCHER.FLANN_CHECKS}
            self.img_processors["flann"] = cv2.FlannBasedMatcher(index_params,
                                                                 search_params)

        # Create a stitcher for individual tiles.
        self.tile_stitcher = TileStitcher(cfg=cfg, img_processors=self.img_processors)

    def compute_neighbor_matches(self, position, position_neigh, img, img_neigh):
        """Compute matches from one tile in a grid to its neighbor. Supports
        all matching directions.

        :param position: Target tile position.
        :param position_neigh: Neighbor tile position.
        :param img: Target image.
        :param img_neigh: Neighbor image.
        :return: Detected matches (target first, neighbor second) and the corresponding
                 confidence scores.
        """
        horizontal = (position_neigh[0] == position[0])
        swapped = (position_neigh[0] < position[0]) or (position_neigh[1] < position[1])

        if swapped:
            matches_neigh, matches, conf = self.tile_stitcher.compute_matches(
                img1_full=img_neigh,
                img2_full=img,
                horizontal=horizontal
            )
        else:
            matches, matches_neigh, conf = self.tile_stitcher.compute_matches(
                img1_full=img,
                img2_full=img_neigh,
                horizontal=horizontal
            )

        return matches, matches_neigh, conf

    def stitch_grid_mst(self, tile_paths):
        """Stitch images in a grid by constructing a minimum spanning tree (MST)
        using Prim-Jarník's algorithm to estimate the best stitching order.

        :param tile_paths: Paths to image tiles in the grid.
        :return: Stitched grid image and the coordinates of the root tile.
        """
        grid_size = tile_paths.shape
        tile_count = grid_size[0] * grid_size[1]
        if tile_count < 2:
            raise ValueError(f"Cannot stitch a single tile: {tile_paths[0, 0]}")

        # Find LoFTR or SIFT keypoint matches between each pair of adjacent image tiles.
        # By default, the algorithm starts from the middle of the grid so that any
        # misalignments are distributed as evenly as possible.
        if self.cfg.STITCHER.ROOT_TILE:
            position = self.cfg.STITCHER.ROOT_TILE
        else:
            position = (grid_size[0] // 2, grid_size[1] // 2)
        node = None
        processed_tiles = 0
        tile_nodes = []
        tile_queue = PriorityQueue()
        tile_states = np.zeros(grid_size, dtype=bool)
        while processed_tiles < tile_count:
            # Update current position.
            if processed_tiles > 0:
                # All tiles neighboring the already processed ones should be already
                # weighted. Select the best one (unless already selected).
                try:
                    best_tile = tile_queue.get(block=False).node
                    while tile_states[best_tile.position]:
                        best_tile = tile_queue.get(block=False).node
                except Empty:
                    break

                best_tile.estimate_transformation()
                tile_nodes.append(best_tile)
                node = best_tile
                position = best_tile.position
            processed_tiles += 1
            tile_states[position] = True

            # Match the current image to its neighbors.
            img = None
            tile_path = tile_paths[position]
            neighbors = self._get_unprocessed_neighbors(position, tile_states)
            for position_neigh in neighbors:
                # Load the images.
                neighbor_path = tile_paths[position_neigh]
                if img is None:
                    img = self.cache.load_img(tile_path)
                    if not tile_nodes:
                        # The first tile is connected to the tree automatically.
                        node = TileNode(cfg=self.cfg, img=img, position=position,
                                        transformation=np.identity(3))
                        tile_nodes.append(node)
                img_neigh = self.cache.load_img(neighbor_path)

                # Compute matches. First image must always be the top or left one.
                matches, matches_neigh, conf = self.compute_neighbor_matches(
                    position=position,
                    position_neigh=position_neigh,
                    img=img,
                    img_neigh=img_neigh
                )

                """
                # TODO: move to utility function.
                # Make a matching figure.

                match_samples = random.sample(range(0, len(matches)),
                                              min(max_matches_shown, len(matches)))
                color = cm.jet(conf[match_samples])
                text = [f"Total Matches: {len(matches)}"]
                fig = make_matching_figure(img, img_neigh, matches[match_samples],
                                        matches_neigh[match_samples], color, text=text)
                out_path = os.path.join(out_dir, f"g{grid_index:04d}",
                                        f"{position_neigh}_matches.png")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                fig.savefig(out_path)
                """

                # Ensure there is enough matches.
                if len(matches) < 3 or (self.cfg.STITCHER.TRANSFORM_TYPE
                                        == "perspective" and len(matches) < 4):
                    print("Not enough matches found for image pair: "
                          f"{(tile_path, neighbor_path)}")
                    continue

                node_neigh = TileNode(
                    cfg=self.cfg,
                    img=img_neigh,
                    position=position_neigh,
                    parent=node,
                    matches=matches_neigh,
                    matches_parent=matches
                )

                # Calculate weight as the negative sum of squared confidence scores.
                weight = -(conf**2).sum()

                # Queue up the neighbor.
                tile_queue.put(WeightedTileNode(weight, node_neigh))

        # Find the boundaries of the final image.
        return self._stitch_mst_nodes(tile_nodes)

    def _get_grid_boundary(self, tile_nodes):
        """Calculate the bounding coordinates of the given image grid after warping.

        :param tile_nodes: Image tiles with computed transformations.
        :return: Bounding coordinates of the grid: (minX, maxX), (minY, maxY).
        """
        # Go over all image tiles (even those not at the grid boundary since
        # uneven grids are supported) and calculate grid boundary coordinates
        # as the minimums and maximums of the image tile warped corners.
        minX, maxX, minY, maxY = np.inf, -np.inf, np.inf, -np.inf
        for tile_node in tile_nodes:
            tile_boundary = tile_node.get_warped_boundary()
            minX = min(minX, tile_boundary[0, 0])
            maxX = max(maxX, tile_boundary[0, 1])
            minY = min(minY, tile_boundary[1, 0])
            maxY = max(maxY, tile_boundary[1, 1])

        return np.array([[minX, maxX],
                         [minY, maxY]])

    def _get_unprocessed_neighbors(self, position, tile_states):
        """Retrieve neighbors of a given tile that have not yet been processed.

        :param position: Target tile position.
        :param tile_states: Array of tile process states.
        :return: List of unprocessed neighbors.
        """
        grid_size = tile_states.shape
        row, column = position

        neighbors = []
        if row > 0 and not tile_states[row - 1, column]:
            neighbors.append((row - 1, column))
        if row < grid_size[0] - 1 and not tile_states[row + 1, column]:
            neighbors.append((row + 1, column))
        if column > 0 and not tile_states[row, column - 1]:
            neighbors.append((row, column - 1))
        if column < grid_size[1] - 1 and not tile_states[row, column + 1]:
            neighbors.append((row, column + 1))

        return neighbors

    def _stitch_mst_nodes(self, tile_nodes):
        """Stitch all tile nodes in a processed MST.

        :param tile_nodes: Image tile nodes representing the MST.
        :return: Final stitched grid image and the coordinates of the root tile.
        """
        # Find the boundaries of the final image.
        grid_boundary = self._get_grid_boundary(tile_nodes)

        # Calculate the corresponding translation matrix and the resulting shape.
        T = np.identity(3)
        if grid_boundary[0, 0] < 0:
            T[0, 2] = -grid_boundary[0, 0]
        if grid_boundary[1, 0] < 0:
            T[1, 2] = -grid_boundary[1, 0]

        result_shape = (round(grid_boundary[0, 1] - grid_boundary[0, 0]),
                        round(grid_boundary[1, 1] - grid_boundary[1, 0]))

        # Stitch the tiles together. Start from the correctly translated root tile.
        if self.cfg.STITCHER.COLORED_OUTPUT:
            tile_nodes[0].color_coat_image()
        stitched_image = cv2.warpPerspective(tile_nodes[0].img, T, result_shape,
                                             flags=cv2.INTER_NEAREST)
        for tile_node in tile_nodes[1:]:
            if self.cfg.STITCHER.COLORED_OUTPUT:
                tile_node.color_coat_image()
            stitched_image = self.tile_stitcher.stitch_tile(stitched_image,
                                                            tile_node, T)
        return stitched_image, (T[0, 2], T[1, 2])
