import cv2
import io
import matplotlib.cm as cm
import numpy as np
import os
import random
import sys
import torch
from graphslam.edge.edge_odometry import EdgeOdometry
from graphslam.graph import Graph
from graphslam.pose.se2 import PoseSE2
from graphslam.util import upper_triangular_matrix_to_full_matrix
from graphslam.vertex import Vertex
from queue import PriorityQueue, Empty
from LoFTR.src.loftr import LoFTR, default_cfg
from skimage.measure import ransac
from skimage.transform import SimilarityTransform, AffineTransform, ProjectiveTransform
from src.pipeline.tile_node import TileNode, WeightedTileNode
from src.pipeline.tile_stitcher import TileStitcher
from LoFTR.src.utils.plotting import make_matching_figure


class GridStitcher:
    """Stitcher for grids of overlapping images."""

    def __init__(self, cfg, img_loader):
        """GridStitcher constructor.

        :param cfg: DEMIS configuration.
        :param img_loader: Image loader.
        """
        self.cfg = cfg
        self.img_loader = img_loader
        self.img_processors = {"loftr": None, "sift": None, "flann": None}

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
            self.img_processors["flann"] = cv2.FlannBasedMatcher(
                index_params, search_params
            )
        else:
            raise ValueError(
                "Invalid matching method: " f"{self.cfg.STITCHER.MATCHING_METHOD}"
            )

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
        horizontal = position_neigh[0] == position[0]
        swapped = (position_neigh[0] < position[0]) or (position_neigh[1] < position[1])

        if swapped:
            matches_neigh, matches, conf = self.tile_stitcher.compute_matches(
                img1_full=img_neigh, img2_full=img, horizontal=horizontal
            )
        else:
            matches, matches_neigh, conf = self.tile_stitcher.compute_matches(
                img1_full=img, img2_full=img_neigh, horizontal=horizontal
            )

        return matches, matches_neigh, conf

    def stitch_grid(self, tile_paths, plot_prefix=""):
        """Stitch images in a grid.

        :param tile_paths: Paths to image tiles in the grid.
        :param plot_prefix: Prefix for plots generated during stitching.
        :return: Stitched grid image and the coordinates of the anchor tile.
        """
        # Run the configured grid stitching algorithm.
        if self.cfg.STITCHER.CONSTRUCTION_METHOD == "mst":
            return self.stitch_grid_mst(tile_paths, plot_prefix)
        elif self.cfg.STITCHER.CONSTRUCTION_METHOD == "slam":
            return self.stitch_grid_slam(tile_paths, plot_prefix)
        else:
            raise ValueError(
                "Invalid grid construction method: "
                f"{self.cfg.STITCHER.CONSTRUCTION_METHOD}"
            )

    def stitch_grid_mst(
        self,
        tile_paths,
        transformations=None,
        root_transformation=None,
        plot_prefix="",
    ):
        """Stitch images in a grid by constructing a minimum spanning tree (MST)
        using Prim-Jarník's algorithm to estimate the best stitching order.

        :param tile_paths: Paths to image tiles in the grid.
        :param transformations: Optional array with precomputed transformations and
                                their scores. If supplied, is will replace homography
                                estimation.
        :param root_transformation: Optional transformation for the root tile.
        :param plot_prefix: Prefix for plots generated during stitching.
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

                if transformations is None:
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
                    img = self.img_loader.load_img(tile_path)
                    if not tile_nodes:
                        # The first tile is connected to the tree automatically.
                        if root_transformation is None:
                            transformation = np.identity(3)
                        else:
                            transformation = root_transformation
                        node = TileNode(
                            cfg=self.cfg,
                            img=img,
                            position=position,
                            transformation=transformation,
                        )
                        tile_nodes.append(node)
                img_neigh = self.img_loader.load_img(neighbor_path)

                if transformations is None:
                    # Compute matches. First image must always be the top or left one.
                    matches, matches_neigh, conf = self.compute_neighbor_matches(
                        position=position,
                        position_neigh=position_neigh,
                        img=img,
                        img_neigh=img_neigh,
                    )

                    # Ensure there is enough matches.
                    if len(matches) < 3 or (
                        self.cfg.STITCHER.TRANSFORM_TYPE == "perspective"
                        and len(matches) < 4
                    ):
                        print(
                            "Not enough matches found for image pair: "
                            f"{(tile_path, neighbor_path)}"
                        )
                        continue

                    # Make matching figure if desired.
                    if self.cfg.STITCHER.SAVE_MATCHES:
                        self._plot_matches(
                            img,
                            img_neigh,
                            matches,
                            matches_neigh,
                            conf,
                            position,
                            position_neigh,
                            self.cfg.STITCHER.SAVE_MATCHES_FRACTION,
                            plot_prefix,
                        )

                    node_neigh = TileNode(
                        cfg=self.cfg,
                        img=img_neigh,
                        position=position_neigh,
                        parent=node,
                        matches=matches_neigh,
                        matches_parent=matches,
                    )

                    # Calculate weight as the negative sum of squared confidence scores.
                    weight = -(conf**2).sum()
                else:
                    # Use precomputed transformations to construct the neighbor node.
                    index = position[0] * grid_size[1] + position[1]
                    index_neigh = position_neigh[0] * grid_size[1] + position_neigh[1]
                    scored_transformation = transformations[index, index_neigh]

                    node_neigh = TileNode(
                        cfg=self.cfg,
                        img=img_neigh,
                        parent=node,
                        position=position_neigh,
                        transformation=scored_transformation[0],
                    )
                    weight = scored_transformation[1]

                # Queue up the neighbor.
                tile_queue.put(WeightedTileNode(weight, node_neigh))

        # Find the boundaries of the final image.
        return self.stitch_mst_nodes(tile_nodes)

    def stitch_grid_slam(self, tile_paths, plot_prefix=""):
        """Stitch images in a grid using SLAM optimisation. Cannot be done when using
        full perspective transformations.

        :param tile_paths: Paths to image tiles in the grid.
        :param plot_prefix: Prefix for plots generated during stitching.
        :return: Stitched grid image and the coordinates of the reference tile.
        """
        if self.cfg.STITCHER.TRANSFORM_TYPE == "perspective":
            raise ValueError(
                "Perspective transformations are not supported when "
                "using SLAM optimisation."
            )

        grid_size = tile_paths.shape
        tile_count = grid_size[0] * grid_size[1]
        if tile_count < 2:
            raise ValueError(f"Cannot stitch a single tile: {tile_paths[0, 0]}")

        # Find LoFTR or SIFT keypoint matches between each pairs of adjacent image
        # tiles from the top left image to the bottom right image. Use the matches
        # to build a SLAM model of the stitched tiles.
        vertices = {}
        edges = {}
        for (row, column), tile_path in np.ndenumerate(tile_paths):
            # Get neighbors in top-to-bottom, left-to-right order.
            neighbors = []
            if row < grid_size[0] - 1:
                neighbors.append((row + 1, column))
            if column < grid_size[1] - 1:
                neighbors.append((row, column + 1))

            img = None
            node = None
            for position_neigh in neighbors:
                # Load the images.
                neighbor_path = tile_paths[position_neigh]
                if img is None:
                    img = self.img_loader.load_img(tile_path)
                img_neigh = self.img_loader.load_img(neighbor_path)

                # Represent neighbor transformations using a tree.
                if node is None:
                    node = TileNode(
                        cfg=self.cfg,
                        img=img,
                        position=(row, column),
                        transformation=np.identity(3),
                    )

                # Compute neighbor matches.
                matches, matches_neigh, conf = self.compute_neighbor_matches(
                    position=(row, column),
                    position_neigh=position_neigh,
                    img=img,
                    img_neigh=img_neigh,
                )

                # Make matching figure if desired.
                if self.cfg.STITCHER.SAVE_MATCHES:
                    self._plot_matches(
                        img,
                        img_neigh,
                        matches,
                        matches_neigh,
                        conf,
                        (row, column),
                        position_neigh,
                        self.cfg.STITCHER.SAVE_MATCHES_FRACTION,
                        plot_prefix,
                    )

                # Ensure there is enough matches.
                if len(matches) < 3:
                    print(
                        "Not enough matches found for image pair: "
                        f"{(tile_path, neighbor_path)}"
                    )
                    continue

                # Calculate the neighbor transformation.
                node_neigh = TileNode(
                    cfg=self.cfg,
                    img=img_neigh,
                    position=position_neigh,
                    parent=node,
                    matches=matches_neigh,
                    matches_parent=matches,
                )
                node_neigh.estimate_transformation()
                pose = node_neigh.remove_scaling()

                # Build vertices with initial position estimates.
                index = row * grid_size[1] + column
                index_neigh = position_neigh[0] * grid_size[1] + position_neigh[1]
                if index not in vertices:
                    vertices[index] = self._get_slam_vertex(
                        img, (row, column), grid_size
                    )
                if index_neigh not in vertices:
                    vertices[index_neigh] = self._get_slam_vertex(
                        img_neigh, position_neigh, grid_size
                    )

                # Add edges with homography-based translation and rotation estimates.
                edge_index = f"{index}_{index_neigh}"
                edges[edge_index] = self._get_slam_edge(index, index_neigh, pose)

        # Build the SLAM graph and optimise.
        slam_graph = Graph(list(edges.values()), list(vertices.values()))
        stdout = sys.stdout
        try:
            sys.stdout = io.StringIO()
            slam_graph.optimize()
        finally:
            sys.stdout = stdout

        # Convert the graph to a simple MST (all nodes connected directly to
        # the root node) and stitch the grid using MST stitching.
        tile_nodes = self._slam_graph_to_mst(vertices, tile_paths)
        return self.stitch_mst_nodes(tile_nodes)

    def stitch_mst_nodes(self, tile_nodes):
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

        result_shape = (
            round(grid_boundary[0, 1] - grid_boundary[0, 0]),
            round(grid_boundary[1, 1] - grid_boundary[1, 0]),
        )

        # Stitch the tiles together. Start from the correctly translated root tile.
        if self.cfg.STITCHER.COLORED_OUTPUT:
            tile_nodes[0].color_coat_image()
        M = T @ tile_nodes[0].transformation
        stitched_image = cv2.warpPerspective(
            tile_nodes[0].img, M, result_shape, flags=cv2.INTER_NEAREST
        )
        for tile_node in tile_nodes[1:]:
            if self.cfg.STITCHER.COLORED_OUTPUT:
                tile_node.color_coat_image()
            stitched_image = self.tile_stitcher.stitch_tile(
                stitched_image, tile_node, T
            )
        return stitched_image, (T[0, 2], T[1, 2])

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

        return np.array([[minX, maxX], [minY, maxY]])

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

    def _get_slam_vertex(self, img, position, grid_size, fixed_first=True):
        """Constructs a new vertex of a SLAM graph in the middle of the given tile.

        :param img: Image tile for which the vertex should be created.
        :param position: Position of the image tile.
        :param grid_size: Size of the stitched grid.
        :param fixed: Whether the vertex position should remain fixed during
                      optimisation when it is the first vertex in the grid.
        :return: New SLAM vertex.
        """
        index = position[0] * grid_size[1] + position[1]
        fixed = fixed_first and index == 0
        pos_x = img.shape[1] // 2 + position[1] * (
            img.shape[1] * (1 - self.cfg.DATASET.TILE_OVERLAP)
        )
        pos_y = img.shape[0] // 2 + position[0] * (
            img.shape[0] * (1 - self.cfg.DATASET.TILE_OVERLAP)
        )
        return Vertex(index, PoseSE2((pos_x, pos_y), 0), fixed=fixed)

    def _get_slam_edge(self, index, index_neigh, pose):
        """Constructs a new SLAM graph edge.

        :param index: Index of the source vertex.
        :param index_neigh: Index of the destination vertex.
        :param pose: Pose associated with the edge.
        :return: New SLAM edge.
        """
        # Create the information matrix.
        information = upper_triangular_matrix_to_full_matrix(
            self.cfg.STITCHER.EDGE_INFORMATION, 3
        )
        return EdgeOdometry(
            vertex_ids=(index, index_neigh),
            information=information,
            estimate=PoseSE2((pose[0], pose[1]), pose[2]),
        )

    def _slam_graph_to_mst(self, vertices, tile_paths):
        """Converts a SLAM graph that represents a grid of stitched images to a MST.

        :param vertices: Indexed dictionary of SLAM graph vertices.
        :param tile_paths: Paths to image tiles in the grid.
        :return: List of image tile nodes representing the constructed MST.
        """
        # Get the starting position.
        grid_size = tile_paths.shape
        if self.cfg.STITCHER.ROOT_TILE:
            position_root = self.cfg.STITCHER.ROOT_TILE
        else:
            position_root = (grid_size[0] // 2, grid_size[1] // 2)

        # Create the root node.
        index = position_root[0] * grid_size[1] + position_root[1]
        pose_root = vertices[index].pose
        root = TileNode(
            cfg=self.cfg,
            img=self.img_loader.load_img(tile_paths[position_root]),
            position=position_root,
            transformation=np.identity(3),
        )
        tile_nodes = [root]

        # Create the remaining nodes and connect them directly to the root noce.
        for position, tile_path in np.ndenumerate(tile_paths):
            if position == position_root:
                continue

            # Load the current image.
            img = self.img_loader.load_img(tile_path)

            # Calculate translation and rotation matrices based on pose difference
            # to root.
            index = position[0] * grid_size[1] + position[1]
            pose_node = vertices[index].pose
            pose_difference = pose_node - pose_root

            T = np.identity(3)
            R = np.identity(3)

            T[0, 2] = pose_difference[0]
            T[1, 2] = pose_difference[1]
            R[0, 0] = np.cos(pose_difference[2])
            R[0, 1] = -np.sin(pose_difference[2])
            R[1, 0] = -R[0, 1]
            R[1, 1] = R[0, 0]

            # Add the neighbor node to the MST.
            node = TileNode(
                cfg=self.cfg,
                img=img,
                parent=root,
                position=position,
                transformation=(T @ R),
            )
            tile_nodes.append(node)

        return tile_nodes

    def _get_slam_transformations(self, vertices, edges, grid_size):
        """Retrieves transformations between adjacent vertices in a SLAM graph. The
        transformations are scored by errors corresponding to the associated edge.

        :param vertices: Indexed dictionary of SLAM graph vertices.
        :param edges: Indexed dictionary of SLAM graph edges.
        :param grid_size: Size of the stitched grid.
        :return: Adjacency matrix of scored SLAM graph transformations.
        """
        # Go over all vertices (and the tiles they represent) from top to bottom and
        # left to right and register their transformations.
        transformations = np.empty((len(vertices), len(vertices)), dtype=object)
        for (row, column) in np.ndindex(grid_size):
            # Get neighbors in top-to-bottom, left-to-right order.
            neighbors = []
            if row < grid_size[0] - 1:
                neighbors.append((row + 1, column))
            if column < grid_size[1] - 1:
                neighbors.append((row, column + 1))

            for position_neigh in neighbors:
                index = row * grid_size[1] + column
                index_neigh = position_neigh[0] * grid_size[1] + position_neigh[1]

                # Calculate translation and rotation matrices based on pose difference.
                pose_difference = vertices[index_neigh].pose - vertices[index].pose

                T = np.identity(3)
                R = np.identity(3)

                T[0, 2] = pose_difference[0]
                T[1, 2] = pose_difference[1]
                R[0, 0] = np.cos(pose_difference[2])
                R[0, 1] = -np.sin(pose_difference[2])
                R[1, 0] = -R[0, 1]
                R[1, 1] = R[0, 0]

                # Calculate score by comparing against the expected transformation.
                # Angles are counted in degrees to increase their weight.
                edge_index = f"{index}_{index_neigh}"
                edge_estimate = edges[edge_index].estimate
                error = abs(pose_difference - edge_estimate)
                error[2] = np.degrees(error[2])
                score = np.sum(np.array(error))

                # Save the transformation and its score.
                M = T @ R
                transformations[index, index_neigh] = (M, score)
                transformations[index_neigh, index] = (np.linalg.inv(M), score)

        return transformations

    def _plot_matches(
        self,
        img,
        img_neigh,
        matches,
        matches_neigh,
        conf,
        position,
        position_neigh,
        fraction=1,
        prefix="",
        color_conf=False,
    ):
        """Plots matches between two tiles.

        :param img: Current image tile.
        :param img_neigh: Neighboring image tile.
        :param matches: Matches of the current image tile.
        :param matches_neigh: Matches of the neighboring image tile.
        :param conf: Match confidence scores.
        :param position: Position of the current image tile.
        :param position_neigh: Position of the neighboring image tile.
        :param fraction: Specifies the fraction of matches to plot.
        :param prefix: Optional output prefix.
        :param color_conf: Whether to color code matches based on confidence scores.
        """
        if position[1] == position_neigh[1]:
            return
        # Get transform settings.
        min_samples = 3
        if self.cfg.STITCHER.TRANSFORM_TYPE == "affine":
            transform_type = AffineTransform
        elif self.cfg.STITCHER.TRANSFORM_TYPE == "partial-affine":
            transform_type = SimilarityTransform
        elif self.cfg.STITCHER.TRANSFORM_TYPE == "perspective":
            transform_type = ProjectiveTransform
            min_samples = 4
        else:
            raise ValueError(
                "Invalid transform type: " f"{self.cfg.STITCHER.TRANSFORM_TYPE}"
            )

        # Apply RANSAC to find inlier keypoints.
        _, inliers_map = ransac(
            (matches, matches_neigh),
            transform_type,
            min_samples=min_samples,
            residual_threshold=self.cfg.STITCHER.RANSAC_THRESHOLD,
            max_trials=100,
        )

        # Select matches to plot.
        inliers = matches[inliers_map]
        inliers_neigh = matches_neigh[inliers_map]
        match_samples = random.sample(
            range(0, len(inliers)), int(len(inliers) * fraction)
        )

        # Prepare color coding.
        if color_conf:
            color = cm.jet(conf[match_samples])
        else:
            color = np.zeros((len(match_samples), 3))
            color[:, 1] = 1

        # Make the matching figure.
        fig_path = os.path.join(
            self.cfg.STITCHER.OUTPUT_PATH,
            f"{prefix}{position}_{position_neigh}_"
            f"{len(inliers)}_"
            f"{self.cfg.STITCHER.MATCHING_METHOD}.png",
        )
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        make_matching_figure(
            img,
            img_neigh,
            inliers[match_samples],
            inliers_neigh[match_samples],
            color,
            dpi=200,
            path=fig_path,
        )
