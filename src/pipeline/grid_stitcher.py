"""Generic stitcher of grids of overlapping images.

Project: Deep Electron Microscopy Image Stitching (DEMIS)
Author: Petr Šilling
Year: 2023
"""

import io
import json
import os
import random
import sys
from queue import Empty, PriorityQueue

import cv2
import lmfit
import matplotlib.cm as cm
import numpy as np
import torch
from graphslam.edge.edge_odometry import EdgeOdometry
from graphslam.graph import Graph
from graphslam.pose.se2 import PoseSE2
from graphslam.util import upper_triangular_matrix_to_full_matrix
from graphslam.vertex import Vertex
from skimage.measure import ransac
from skimage.transform import AffineTransform, EuclideanTransform, ProjectiveTransform, SimilarityTransform
from torchvision.models.optical_flow import raft_small as raft

from LoFTR.src.loftr import LoFTR, default_cfg
from LoFTR.src.utils.plotting import make_matching_figure
from src.pipeline.tile_node import TileNode, WeightedTileNode
from src.pipeline.tile_stitcher import TileStitcher


class GridStitcher:
    """Stitcher for grids of overlapping images."""

    def __init__(self, cfg, img_loader):
        """GridStitcher constructor.

        :param cfg: DEMIS configuration.
        :param img_loader: Image loader.
        """
        self.i = 0
        self.cfg = cfg
        self.img_loader = img_loader
        self.img_processors = {
            "loftr": None,
            "sift": None,
            "surf": None,
            "orb": None,
            "bfmatcher": None,
            "flann": None,
            "raft": None,
        }

        # Setup the matcher instance.
        if cfg.STITCHER.MATCHING_METHOD == "loftr":
            # Prepare LoFTR for matching using outdoor weights (better than indoor).
            self.img_processors["loftr"] = LoFTR(config=default_cfg)
            self.img_processors["loftr"].load_state_dict(torch.load(cfg.LOFTR.CHECKPOINT_PATH)["state_dict"])
            self.img_processors["loftr"] = self.img_processors["loftr"].eval().cuda()
        elif cfg.STITCHER.MATCHING_METHOD in ("sift", "surf"):
            if cfg.STITCHER.MATCHING_METHOD == "sift":
                self.img_processors["sift"] = cv2.SIFT_create()
            else:
                # SURF will not function with standard OpenCV distribution due to patents.
                self.img_processors["surf"] = cv2.xfeatures2d.SURF_create(cfg.STITCHER.HESSIAN_THRESHOLD)

            index_params = {"algorithm": 1, "trees": cfg.STITCHER.FLANN_TREES}
            search_params = {"checks": cfg.STITCHER.FLANN_CHECKS}
            self.img_processors["flann"] = cv2.FlannBasedMatcher(index_params, search_params)
        elif cfg.STITCHER.MATCHING_METHOD == "orb":
            self.img_processors["orb"] = cv2.ORB_create()
            self.img_processors["bfmatcher"] = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            raise ValueError("Invalid matching method: " f"{self.cfg.STITCHER.MATCHING_METHOD}")

        # Create a stitcher for individual tiles.
        self.tile_stitcher = TileStitcher(cfg=cfg, img_processors=self.img_processors)

        # Create RAFT model instance for optical flow refinement.
        if cfg.STITCHER.OPTICAL_FLOW_REFINEMENT:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.img_processors["raft"] = raft(pretrained=True, progress=False).to(self.device).eval()

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

        if self.cfg.STITCHER.MAX_MATCHES:
            # Sort by confidence and remove bad matches.
            conf_indices = conf.argsort()[::-1]
            matches = matches[conf_indices][: self.cfg.STITCHER.MAX_MATCHES]
            matches_neigh = matches_neigh[conf_indices][: self.cfg.STITCHER.MAX_MATCHES]
            conf = conf[conf_indices][: self.cfg.STITCHER.MAX_MATCHES]

        if self.cfg.STITCHER.MIN_MATCH_CONFIDENCE:
            good_indices = conf > self.cfg.STITCHER.MIN_MATCH_CONFIDENCE
            if good_indices.sum() < self.cfg.STITCHER.MIN_MATCHES_AFTER_CONFIDENCE_TEST:
                conf_indices = conf.argsort()[::-1]
                matches = matches[conf_indices][: self.cfg.STITCHER.MIN_MATCHES_AFTER_CONFIDENCE_TEST]
                matches_neigh = matches_neigh[conf_indices][: self.cfg.STITCHER.MIN_MATCHES_AFTER_CONFIDENCE_TEST]
                conf = conf[conf_indices][: self.cfg.STITCHER.MIN_MATCHES_AFTER_CONFIDENCE_TEST]
            else:
                matches = matches[good_indices]
                matches_neigh = matches_neigh[good_indices]
                conf = conf[good_indices]

        return matches, matches_neigh, conf

    def stitch_grid(self, tile_paths, transformations_only=False, plot_prefix=""):
        """Stitch images in a grid.

        :param tile_paths: Paths to image tiles in the grid.
        :param transformations_only: Instead of stitching the image internally, only return
        the global transformations for each tile in the grid.
        :param plot_prefix: Prefix for plots generated during stitching.
        :return: Stitched grid image and the coordinates of the anchor tile.
        """
        # Run the configured grid stitching algorithm.
        if self.cfg.STITCHER.CONSTRUCTION_METHOD == "mst":
            return self.stitch_grid_mst(tile_paths, transformations_only=transformations_only, plot_prefix=plot_prefix)
        elif self.cfg.STITCHER.CONSTRUCTION_METHOD == "slam":
            return self.stitch_grid_slam(tile_paths, transformations_only, plot_prefix)
        elif self.cfg.STITCHER.CONSTRUCTION_METHOD == "optimised":
            return self.stitch_grid_optimised(tile_paths, transformations_only, plot_prefix)
        else:
            raise ValueError("Invalid grid construction method: " f"{self.cfg.STITCHER.CONSTRUCTION_METHOD}")

    def stitch_grid_optimised(self, tile_paths, transformations_only=False, plot_prefix=""):
        """Stitch images in a grid using global least squares optimisation. Supports translation,
        euclidean, affine, and affine with radial/tangential deformations.

        :param tile_paths: Paths to image tiles in the grid.
        :param transformations_only: Instead of stitching the image internally, only return
        the global transformations for each tile in the grid.
        :param plot_prefix: Prefix for plots generated during stitching.
        :return: Stitched grid image and the coordinates of the reference tile.
        """
        if self.cfg.STITCHER.TRANSFORM_TYPE not in ("translation", "euclidean", "affine", "affine_radial"):
            raise ValueError(
                f"Transform type '{self.cfg.STITCHER.TRANSFORM_TYPE}' not supported for the selected method. "
                "Please choose one of: translation, euclidean, affine, affine_radial."
            )

        grid_size = tile_paths.shape
        tile_count = grid_size[0] * grid_size[1]
        if tile_count < 2:
            raise ValueError(f"Cannot stitch a single tile: {tile_paths[0, 0]}")

        # Find keypoint matches between each pairs of adjacent image
        # tiles from the top left image to the bottom right image.
        shape = None
        total_match_count = 0
        indexed_matches = {}
        initial_params = lmfit.Parameters()
        for position, tile_path in np.ndenumerate(tile_paths):
            # Get neighbors in top-to-bottom, left-to-right order.
            neighbors = []
            row, column = position
            if row < grid_size[0] - 1:
                neighbors.append((row + 1, column))
            if column < grid_size[1] - 1:
                neighbors.append((row, column + 1))

            img = None
            for position_neigh in neighbors:
                # Load the images.
                neighbor_path = tile_paths[position_neigh]
                if img is None:
                    img = self.img_loader.load_img(tile_path)
                img_neigh = self.img_loader.load_img(neighbor_path)

                # Compute neighbor matches.
                matches, matches_neigh, conf = self.compute_neighbor_matches(
                    position=(row, column), position_neigh=position_neigh, img=img, img_neigh=img_neigh
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
                        self.cfg.STITCHER.SAVE_MATCHES_COLOR_CODED,
                    )

                if self.cfg.STITCHER.SAVE_PAIRWISE_IMAGES:
                    self._save_pairwise_image(
                        img,
                        img_neigh,
                        matches,
                        matches_neigh,
                        (row, column),
                        position_neigh,
                        plot_prefix,
                    )

                # Ensure there is enough matches.
                match_count = len(matches)
                total_match_count += match_count
                if match_count < 3:
                    print("Not enough matches found for image pair: " f"{(tile_path, neighbor_path)}")
                    continue

                # Build vertices with initial position estimates.
                index = position[0] * grid_size[1] + position[1]
                index_neigh = position_neigh[0] * grid_size[1] + position_neigh[1]
                self._add_image_params(initial_params, img.shape, index, position)
                self._add_image_params(initial_params, img.shape, index_neigh, position_neigh)
                shape = img.shape

                indexed_matches[(index, index_neigh)] = (np.stack(matches), np.stack(matches_neigh), np.stack(conf))

        # Optimise tile poses to minimise global reprojection errors.
        result = lmfit.minimize(
            fcn=self._calculate_residuals,
            params=initial_params,
            method="leastsq",
            args=(indexed_matches, total_match_count, shape),
        )

        # Print final parameters.
        # print("Final parameters:")
        # lmfit.report_fit(result.params)

        # Convert the graph to a simple MST (all nodes connected directly to the root node).
        # The MST can then be usedto stitch the grid.
        tile_nodes = self._optimised_result_to_mst(result.params, tile_paths)

        # If optical flow refinement is disabled, return the stitched grid.
        if not self.cfg.STITCHER.OPTICAL_FLOW_REFINEMENT:
            return self.stitch_mst_nodes(tile_nodes, transformations_only)

        # # Debug: Visualize original matches in the coarsely stitched image
        # coarse_stitched_img, _ = self.stitch_mst_nodes(tile_nodes, False, no_color=True)
        # debug_img = (
        #     cv2.cvtColor(coarse_stitched_img.copy(), cv2.COLOR_GRAY2BGR)
        #     if coarse_stitched_img.ndim == 2
        #     else coarse_stitched_img.copy()
        # )
        # for (index, index_neigh), (points1, points2, _) in indexed_matches.items():
        #     # Draw lines between corresponding points
        #     for pt1, pt2 in zip(points1, points2):
        #         pt1_int = tuple(np.round(pt1).astype(int))
        #         pt2_int = tuple(np.round(pt2).astype(int))
        #         cv2.circle(debug_img, pt1_int, 2, (0, 255, 0), -1)
        #         cv2.line(debug_img, pt1_int, pt2_int, (0, 0, 255), 1)
        # cv2.imwrite("output/stitched_original_matches.png", debug_img)

        # Calculate optical flow for each overlapping pair (EMSIQA-FLOW style).
        i = 0
        node_map = {n.position: n for n in tile_nodes}
        refinement_indexed_matches = {}
        refinement_match_count = 0
        for position, _ in np.ndenumerate(tile_paths):
            # Get neighbors in top-to-bottom, left-to-right order.
            neighbors = []
            row, column = position
            if row < grid_size[0] - 1:
                neighbors.append((row + 1, column))
            if column < grid_size[1] - 1:
                neighbors.append((row, column + 1))

            node = node_map[position]
            img = node.img
            transformation = node.transformation
            for position_neigh in neighbors:
                node_neigh = node_map[position_neigh]
                img_neigh = node_neigh.img
                transformation_neigh = node_neigh.transformation

                # Calculate the relative transformation between the tiles.
                transformation_relative = np.linalg.inv(transformation) @ transformation_neigh

                # Create a stitching tree for the adjacent pairs.
                root = TileNode(cfg=self.cfg, img=img, position=position, transformation=np.identity(3))
                leaf = TileNode(
                    cfg=self.cfg, img=img_neigh, position=position_neigh, transformation=transformation_relative
                )
                current_nodes = [root, leaf]

                # Warp the images and retrieve the overlap region.
                warped_images, warped_masks, _ = self.stitch_mst_nodes(
                    current_nodes, separate=True, masks=True, no_color=True
                )

                overlap_regions, overlap_mask, overlap_coords = self.get_overlap_region(
                    warped_images, warped_masks, cropped=True, location=True
                )
                if overlap_regions[0].size == 0 or overlap_regions[1].size == 0:
                    continue  # No overlap region found.

                # Pad overlap regions to next multiple of 8 (needed for RAFT).
                region_h, region_w = overlap_mask.shape
                pad_h = (8 - region_h % 8) % 8
                pad_w = (8 - region_w % 8) % 8
                pad = ((0, pad_h), (0, pad_w))
                region1 = np.pad(overlap_regions[0], pad, mode="constant", constant_values=0)
                region2 = np.pad(overlap_regions[1], pad, mode="constant", constant_values=0)

                # Prepare batched tensors for RAFT.
                region1_tensor = (torch.tensor(region1).to(self.device, dtype=torch.float32) / 255) * 2 - 1
                region2_tensor = (torch.tensor(region2).to(self.device, dtype=torch.float32) / 255) * 2 - 1
                region1_batch = torch.stack(3 * [region1_tensor]).unsqueeze(0)
                region2_batch = torch.stack(3 * [region2_tensor]).unsqueeze(0)

                # Predict optical flow.
                with torch.no_grad():
                    flow = self.img_processors["raft"](region1_batch, region2_batch)[-1]  # (N, 2, H, W)
                flow = flow[0].permute(1, 2, 0).cpu().detach().numpy()  # (H, W, 2)
                flow = flow[:region_h, :region_w]
                del region1_tensor, region2_tensor, region1_batch, region2_batch
                torch.cuda.empty_cache()

                if self.cfg.STITCHER.OPTICAL_FLOW_REFINEMENT_TYPE == "mean":
                    # Find the corresponding keypoint matches for this tile pair.
                    idx1 = position[0] * grid_size[1] + position[1]
                    idx2 = position_neigh[0] * grid_size[1] + position_neigh[1]
                    points1, points2, conf = indexed_matches[(idx1, idx2)]

                    # Add the mean optical flow vector to all keypoint positions.
                    mean_flow = np.mean(flow[overlap_mask > 0], axis=0)
                    points2_new = points2 + mean_flow
                    refinement_indexed_matches[(idx1, idx2)] = (points1, points2_new, conf)
                    refinement_match_count += len(points1)
                elif self.cfg.STITCHER.OPTICAL_FLOW_REFINEMENT_TYPE == "grid":
                    # Sample a grid of points in the overlap region (overlap region coordinates)
                    min_y, _, min_x, _ = overlap_coords
                    grid_spacing = 32  # pixels, can be parameterized
                    region_h, region_w = overlap_mask.shape
                    y_coords = np.arange(0, region_h, grid_spacing)
                    x_coords = np.arange(0, region_w, grid_spacing)
                    grid_points = np.array(np.meshgrid(x_coords, y_coords)).reshape(2, -1).T  # (N, 2), (x, y)

                    # Only keep points where mask is nonzero (mask is in overlap region coordinates)
                    valid_mask = overlap_mask[grid_points[:, 1], grid_points[:, 0]] > 0
                    valid_points = grid_points[valid_mask]

                    # Apply the optical flow to the valid sampled points.
                    flow_vectors = flow[valid_points[:, 1], valid_points[:, 0]]  # (N, 2)
                    # Move points to full-image coordinates
                    points1 = (valid_points + np.array([min_x, min_y])).astype(np.float32)
                    points2 = (valid_points + flow_vectors + np.array([min_x, min_y])).astype(np.float32)

                    # Find the boundaries of the two stitched tiles the corresponding inverse translation matrix.
                    grid_boundary = self._get_grid_boundary(current_nodes)

                    T = np.identity(3)
                    if grid_boundary[0, 0] < 0:
                        T[0, 2] = grid_boundary[0, 0]
                    if grid_boundary[1, 0] < 0:
                        T[1, 2] = grid_boundary[1, 0]

                    # Revert all transformations.
                    M = np.linalg.inv(transformation_relative) @ T
                    points1 = cv2.perspectiveTransform(points1.reshape(-1, 1, 2), T).reshape(-1, 2)
                    points2 = cv2.perspectiveTransform(points2.reshape(-1, 1, 2), M).reshape(-1, 2)

                    # # Debug: Visualize the grid points on the warped_images.
                    # debug_img = warped_images[0].copy()
                    # for pt in points1:
                    #     pt_int = tuple(np.round(pt).astype(int))
                    #     cv2.circle(debug_img, pt_int, 2, (255, 255, 255), -1)
                    # cv2.imwrite(f"output/g{i}_s{i}_{node.position}_{node_neigh.position}_grid_points_0.png", debug_img)

                    # debug_img = warped_images[1].copy()
                    # for pt in points2:
                    #     pt_int = tuple(np.round(pt).astype(int))
                    #     cv2.circle(debug_img, pt_int, 2, (255, 255, 255), -1)
                    # cv2.imwrite(f"output/g{i}_s{i}_{node.position}_{node_neigh.position}_grid_points_1.png", debug_img)

                    # Store the refined matches in 'refinement_indexed_matches' dictionary.
                    idx1 = position[0] * grid_size[1] + position[1]
                    idx2 = position_neigh[0] * grid_size[1] + position_neigh[1]
                    conf = np.ones(len(points1))  # Confidence is set to 1 for all points.
                    refinement_indexed_matches[(idx1, idx2)] = (points1, points2, conf)
                    refinement_match_count += len(points1)

        # Rerun the global optimisation with refined matches.
        result_refined = lmfit.minimize(
            fcn=self._calculate_residuals,
            params=initial_params,
            method="leastsq",
            args=(refinement_indexed_matches, refinement_match_count, shape),
        )
        tile_nodes_refined = self._optimised_result_to_mst(result_refined.params, tile_paths)

        # # Debug: Visualize original matches in the finely stitched image
        # fine_stitched_img, _ = self.stitch_mst_nodes(tile_nodes_refined, False, no_color=True)
        # debug_img = (
        #     cv2.cvtColor(fine_stitched_img.copy(), cv2.COLOR_GRAY2BGR)
        #     if fine_stitched_img.ndim == 2
        #     else fine_stitched_img.copy()
        # )
        # for (index, index_neigh), (points1, points2, _) in refinement_indexed_matches.items():
        #     # Draw lines between corresponding points
        #     for pt1, pt2 in zip(points1, points2):
        #         pt1_int = tuple(np.round(pt1).astype(int))
        #         pt2_int = tuple(np.round(pt2).astype(int))
        #         cv2.circle(debug_img, pt1_int, 2, (0, 255, 0), -1)
        #         cv2.line(debug_img, pt1_int, pt2_int, (0, 0, 255), 1)
        # cv2.imwrite("output/stitched_refined_matches.png", debug_img)

        return self.stitch_mst_nodes(tile_nodes_refined, transformations_only)

    def stitch_grid_slam(self, tile_paths, transformations_only=False, plot_prefix=""):
        """Stitch images in a grid using SLAM optimisation. Cannot be done when using
        full perspective transformations.

        :param tile_paths: Paths to image tiles in the grid.
        :param transformations_only: Instead of stitching the image internally, only return
        the global transformations for each tile in the grid.
        :param plot_prefix: Prefix for plots generated during stitching.
        :return: Stitched grid image and the coordinates of the reference tile.
        """
        if self.cfg.STITCHER.TRANSFORM_TYPE != "euclidean":
            raise ValueError(
                f"Transform type '{self.cfg.STITCHER.TRANSFORM_TYPE}' not supported for the selected method. "
                "Please select the euclidean transform type."
            )

        grid_size = tile_paths.shape
        tile_count = grid_size[0] * grid_size[1]
        if tile_count < 2:
            raise ValueError(f"Cannot stitch a single tile: {tile_paths[0, 0]}")

        # Find keypoint matches between each pairs of adjacent image tiles from the top left image
        # to the bottom right image. Use the matches to build a SLAM model of the stitched tiles.
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
                    node = TileNode(cfg=self.cfg, img=img, position=(row, column), transformation=np.identity(3))

                # Compute neighbor matches.
                matches, matches_neigh, conf = self.compute_neighbor_matches(
                    position=(row, column), position_neigh=position_neigh, img=img, img_neigh=img_neigh
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
                        self.cfg.STITCHER.SAVE_MATCHES_COLOR_CODED,
                    )

                if self.cfg.STITCHER.SAVE_PAIRWISE_IMAGES:
                    self._save_pairwise_image(
                        img,
                        img_neigh,
                        matches,
                        matches_neigh,
                        (row, column),
                        position_neigh,
                        plot_prefix,
                    )

                # Ensure there is enough matches.
                if len(matches) < 3:
                    print("Not enough matches found for image pair: " f"{(tile_path, neighbor_path)}")
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
                pose = node_neigh.get_pose()

                # Build vertices with initial position estimates.
                index = row * grid_size[1] + column
                index_neigh = position_neigh[0] * grid_size[1] + position_neigh[1]
                if index not in vertices:
                    vertices[index] = self._get_slam_vertex(img, (row, column), grid_size)
                if index_neigh not in vertices:
                    vertices[index_neigh] = self._get_slam_vertex(img_neigh, position_neigh, grid_size)

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
        return self.stitch_mst_nodes(tile_nodes, transformations_only)

    def stitch_grid_mst(
        self,
        tile_paths,
        transformations_only=False,
        transformations=None,
        root_transformation=None,
        plot_prefix="",
    ):
        """Stitch images in a grid by constructing a minimum spanning tree (MST)
        using Prim-Jarník's algorithm to estimate the best stitching order.

        :param tile_paths: Paths to image tiles in the grid.
        :param transformations_only: Instead of stitching the image internally, only return
        the global transformations for each tile in the grid.
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

        # Find keypoint matches between each pair of adjacent image tiles. By default, the algorithm
        # starts from the middle of the grid so that any misalignments are distributed as evenly as possible.
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
                        transformation = np.identity(3) if root_transformation is None else root_transformation
                        node = TileNode(cfg=self.cfg, img=img, position=position, transformation=transformation)
                        tile_nodes.append(node)
                img_neigh = self.img_loader.load_img(neighbor_path)

                if transformations is None:
                    # Compute matches. First image must always be the top or left one.
                    matches, matches_neigh, conf = self.compute_neighbor_matches(
                        position=position, position_neigh=position_neigh, img=img, img_neigh=img_neigh
                    )

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
                            self.cfg.STITCHER.SAVE_MATCHES_COLOR_CODED,
                        )

                    if self.cfg.STITCHER.SAVE_PAIRWISE_IMAGES:
                        self._save_pairwise_image(
                            img,
                            img_neigh,
                            matches,
                            matches_neigh,
                            position,
                            position_neigh,
                            plot_prefix,
                        )

                    # Ensure there is enough matches.
                    if len(matches) < 3 or (self.cfg.STITCHER.TRANSFORM_TYPE == "projective" and len(matches) < 4):
                        print("Not enough matches found for image pair: " f"{(tile_path, neighbor_path)}")
                        continue

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
        return self.stitch_mst_nodes(tile_nodes, transformations_only)

    def stitch_mst_nodes(self, tile_nodes, transformations_only=False, separate=False, masks=False, no_color=False):
        """Stitch all tile nodes in a processed MST.

        :param tile_nodes: Image tile nodes representing the MST.
        :param transformations_only: Instead of stitching the image internally, only return
        the global transformations for each tile in the grid.
        :param separate: Whether to output the stitched images separately.
        :param masks: Whether to output masks of each stitched image. Applies to separate image output only.
        :param no_color: If True, do not apply color coating to the stitched images (regardless of configuration).
        :return: Final stitched image (or a list of images if separate output) and the coordinates of the root tile.
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

        # Apply color coating.
        if self.cfg.STITCHER.COLORED_OUTPUT and not no_color:
            for tile_node in tile_nodes:
                tile_node.color_coat_image()

        root_coordinates = (T[0, 2], T[1, 2])
        if transformations_only:
            # Only return the global transformations.
            tile_transformations = {}
            for tile_node in tile_nodes:
                M = T @ tile_node.transformation
                tile_transformations[tile_node.position] = M
            return tile_transformations

        if separate:
            # Stitch tiles separately. The final shape will be the same but only one tile will be present on each image.
            warped_images = []
            warped_masks = []
            for tile_node in tile_nodes:
                M = T @ tile_node.transformation
                stitched_image = cv2.warpPerspective(tile_node.img, M, result_shape, flags=cv2.INTER_NEAREST)
                warped_images.append(stitched_image)

                if masks:
                    mask = np.full_like(tile_node.img, 255)
                    stitched_mask = cv2.warpPerspective(mask, M, result_shape, flags=cv2.INTER_NEAREST)
                    warped_masks.append(stitched_mask)

            if masks:
                return warped_images, warped_masks, root_coordinates
            return warped_images, root_coordinates

        # Stitch tiles together on a single image.
        M = T @ tile_nodes[0].transformation
        stitched_image = cv2.warpPerspective(tile_nodes[0].img, M, result_shape, flags=cv2.INTER_NEAREST)
        for tile_node in tile_nodes[1:]:
            stitched_image = self.tile_stitcher.stitch_tile(stitched_image, tile_node, T)

        return stitched_image, root_coordinates

    def load_transformations(self, path):
        """Load global tile transformation matrices from a JSON file.

        :param path: Path to the JSON file to read.
        :return: Loaded global tile transformations.
        """
        transformations = {}
        with open(path) as json_file:
            json_transformations = json.load(json_file)

            # Parse individual transformations.
            for transformation_dict in json_transformations:
                position = tuple(transformation_dict["position"])
                tile_transformation = np.array(transformation_dict["transformation"], dtype=np.float32)
                transformations[position] = tile_transformation

        return transformations

    def get_overlap_region(self, warped_images, warped_masks, mask=True, cropped=False, location=False):
        """Retrieve the overlapping regions of selected image tiles.

        :param warped_images: Overlapping image tiles warped to the same coordinate space.
        :param warped_masks: Masks for the input image tiles.
        :param mask: Whether to also return the overlapping region mask.
        :param cropped: Whether to crop the result.
        :param location: Whether to also return the (min_y, max_y, min_x, max_x) coordinates of the overlap region.
        :return: The overlap regions of each input image and, possibly, the shared mask and location.
        """
        # Find the shared overlap region.
        overlap_mask = warped_masks[0]
        for img in warped_masks[1:]:
            overlap_mask = cv2.bitwise_and(overlap_mask, img)

        # Apply the shared region mask to each image.
        overlap_regions = []
        nonzero_y, nonzero_x = np.nonzero(overlap_mask)
        min_y, max_y = np.min(nonzero_y), np.max(nonzero_y)
        min_x, max_x = np.min(nonzero_x), np.max(nonzero_x)
        for img in warped_images:
            overlap_img = cv2.bitwise_and(overlap_mask, img)
            if cropped:
                overlap_img = overlap_img[min_y:max_y, min_x:max_x]

            overlap_regions.append(overlap_img)

        results = [overlap_regions]
        if mask:
            if cropped:
                overlap_mask = overlap_mask[min_y:max_y, min_x:max_x]
            results.append(overlap_mask)

        if location:
            results.append((min_y, max_y, min_x, max_x))

        if len(results) == 1:
            return results[0]

        return tuple(results)

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

    def _add_image_params(self, params, shape, index, position, fixed_first=True):
        """Registers global optimisation parameters for a single tile.

        :param params: Parameter set to update.
        :param shape: Shape of image tiles.
        :param index: Index of the target tile.
        :param position: Position of the tile in the stitched grid.
        :param fixed_first: Whether the parameters of the first tile should remain fixed during optimisation.
        """
        x_name = f"img_{index}_x"
        y_name = f"img_{index}_y"
        angle_name = f"img_{index}_angle"
        scale_x_name = f"img_{index}_scale_x"
        scale_y_name = f"img_{index}_scale_y"
        shear_x_name = f"img_{index}_shear_x"
        shear_y_name = f"img_{index}_shear_y"
        k1_name = f"img_{index}_k1"
        k2_name = f"img_{index}_k2"
        k3_name = f"img_{index}_k3"
        p1_name = f"img_{index}_p1"
        p2_name = f"img_{index}_p2"
        if x_name in params:
            return

        x = position[1] * (shape[1] * (1 - self.cfg.DATASET.TILE_OVERLAP))
        y = position[0] * (shape[0] * (1 - self.cfg.DATASET.TILE_OVERLAP))
        vary = fixed_first and index != 0
        angle_vary = vary and self.cfg.STITCHER.TRANSFORM_TYPE in ("euclidean", "affine", "affine_radial")
        affine = self.cfg.STITCHER.TRANSFORM_TYPE in ("affine", "affine_radial")
        radial = self.cfg.STITCHER.TRANSFORM_TYPE == "affine_radial"

        params.add(x_name, value=x, vary=vary, min=x - shape[1], max=x + shape[1])
        params.add(y_name, value=y, vary=vary, min=y - shape[0], max=y + shape[0])
        params.add(angle_name, value=0, vary=angle_vary, min=-np.pi / 6, max=np.pi / 6)
        if affine:
            params.add(scale_x_name, value=1.0, vary=vary, min=0.98, max=1.02)
            params.add(scale_y_name, value=1.0, vary=vary, min=0.98, max=1.02)
            params.add(shear_x_name, value=0.0, vary=vary, min=-0.2, max=0.2)
            params.add(shear_y_name, value=0.0, vary=vary, min=-0.2, max=0.2)
        if radial:
            params.add(k1_name, value=0.0, vary=vary, min=-1e-4, max=1e-4)
            params.add(k2_name, value=0.0, vary=vary, min=-1e-6, max=1e-6)
            params.add(k3_name, value=0.0, vary=vary, min=-1e-8, max=1e-8)
            params.add(p1_name, value=0.0, vary=vary, min=-1e-5, max=1e-5)
            params.add(p2_name, value=0.0, vary=vary, min=-1e-5, max=1e-5)

    def affine_matrix(self, x, y, angle, scale_x, scale_y, shear_x, shear_y):
        # 3x3 matrices for full homogeneous transform
        S = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])

        Sh = np.array([[1, shear_x, 0], [shear_y, 1, 0], [0, 0, 1]])

        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

        T = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])

        # Compose the final transformation
        A = T @ R @ Sh @ S
        return A

    def _transform_points(self, points, pose, affine_params=None, radial_params=None, shape=None):
        """Transform a batch of points according to given pose parameters, affine, and distortion.

        :param points: Points to transform.
        :param pose: Pose to use for the transformation.
        :param affine_params: (scale_x, scale_y, shear) if present.
        :param radial_params: (k1, k2, p1, p2) if present.
        :param shape: Shape of image tiles.
        :return: Transformed points.
        """
        x, y, angle = pose

        # Get affine parameters.
        scale_x, scale_y, shear_x, shear_y = 1.0, 1.0, 0.0, 0.0
        if affine_params is not None:
            scale_x, scale_y, shear_x, shear_y = affine_params

        A = self.affine_matrix(x, y, angle, scale_x, scale_y, shear_x, shear_y)

        # # If A is not the same as A_, print.
        # if A.shape == (3, 3) and not np.allclose(A, A_):

        #     A_ = np.array(
        #         [
        #             [scale_x * np.cos(angle) + shear_x, -np.sin(angle) + shear_y, x],
        #             [np.sin(angle) + shear_x, scale_y * np.cos(angle) + shear_y, y],
        #             [0, 0, 1],
        #         ]
        #     )

        #     self.i += 1
        #     print(A)
        #     print()

        #     # Build a full affine matrix.
        #     print(A_)
        #     if self.i > 100:
        #         stop

        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        points_transformed = (A @ points_homogeneous.T).T[:, :2]

        # Add radial/tangential distortion.
        if radial_params is not None:
            k1, k2, k3, p1, p2 = radial_params
            cx = shape[1] / 2
            cy = shape[0] / 2

            x_ = points_transformed[:, 0] - cx
            y_ = points_transformed[:, 1] - cy
            r2 = x_**2 + y_**2
            r4 = r2**2
            r6 = r2**3

            radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
            x_dist = x_ * radial + 2 * p1 * x_ * y_ + p2 * (r2 + 2 * x_**2)
            y_dist = y_ * radial + p1 * (r2 + 2 * y_**2) + 2 * p2 * x_ * y_

            points_transformed[:, 0] = x_dist + cx
            points_transformed[:, 1] = y_dist + cy

        return points_transformed

    def _calculate_residuals(self, params, indexed_matches, total_match_count, shape):
        """Calculates residuals of transformed points.

        :param params: Optimised parameters.
        :param indexed_matches: Matches indexed by image tile pairs.
        :param total_match_count: Total number of matches.
        :param shape: Shape of image tiles.
        :return: Calculated residuals.
        """
        residuals_index = 0
        residuals = np.empty((2 * total_match_count,), dtype=np.float64)
        for (index, index_neigh), (points, points_neigh, confs) in indexed_matches.items():
            pose = (
                params[f"img_{index}_x"].value,
                params[f"img_{index}_y"].value,
                params[f"img_{index}_angle"].value,
            )
            pose_neigh = (
                params[f"img_{index_neigh}_x"].value,
                params[f"img_{index_neigh}_y"].value,
                params[f"img_{index_neigh}_angle"].value,
            )

            affine = self.cfg.STITCHER.TRANSFORM_TYPE in ("affine", "affine_radial")
            affine_params = None
            affine_params_neigh = None
            if affine:
                affine_params = (
                    params[f"img_{index}_scale_x"].value,
                    params[f"img_{index}_scale_y"].value,
                    params[f"img_{index}_shear_x"].value,
                    params[f"img_{index}_shear_y"].value,
                )
                affine_params_neigh = (
                    params[f"img_{index_neigh}_scale_x"].value,
                    params[f"img_{index_neigh}_scale_y"].value,
                    params[f"img_{index_neigh}_shear_x"].value,
                    params[f"img_{index_neigh}_shear_y"].value,
                )

            radial = self.cfg.STITCHER.TRANSFORM_TYPE == "affine_radial"
            radial_params = None
            radial_params_neigh = None
            if radial:
                radial_params = (
                    params[f"img_{index}_k1"].value,
                    params[f"img_{index}_k2"].value,
                    params[f"img_{index}_k3"].value,
                    params[f"img_{index}_p1"].value,
                    params[f"img_{index}_p2"].value,
                )
                radial_params_neigh = (
                    params[f"img_{index_neigh}_k1"].value,
                    params[f"img_{index_neigh}_k2"].value,
                    params[f"img_{index_neigh}_k3"].value,
                    params[f"img_{index_neigh}_p1"].value,
                    params[f"img_{index_neigh}_p2"].value,
                )

            # Transform points based on current poses, affine transform, and radial/tangential distortion.
            transformed_points = self._transform_points(
                points,
                pose,
                affine_params,
                radial_params,
                shape,
            )
            transformed_points_neigh = self._transform_points(
                points_neigh,
                pose_neigh,
                affine_params_neigh,
                radial_params_neigh,
                shape,
            )

            # Calculate residuals weighted by confidence.
            weighted_residuals = (confs[:, np.newaxis] * (transformed_points_neigh - transformed_points)).flatten()
            next_residuals_index = residuals_index + weighted_residuals.shape[0]
            residuals[residuals_index:next_residuals_index] = weighted_residuals
            residuals_index = next_residuals_index

        return residuals

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
        pos_x = img.shape[1] // 2 + position[1] * (img.shape[1] * (1 - self.cfg.DATASET.TILE_OVERLAP))
        pos_y = img.shape[0] // 2 + position[0] * (img.shape[0] * (1 - self.cfg.DATASET.TILE_OVERLAP))
        return Vertex(index, PoseSE2((pos_x, pos_y), 0), fixed=fixed)

    def _get_slam_edge(self, index, index_neigh, pose):
        """Constructs a new SLAM graph edge.

        :param index: Index of the source vertex.
        :param index_neigh: Index of the destination vertex.
        :param pose: Pose associated with the edge.
        :return: New SLAM edge.
        """
        # Create the information matrix.
        information = upper_triangular_matrix_to_full_matrix(self.cfg.STITCHER.EDGE_INFORMATION, 3)
        return EdgeOdometry(
            vertex_ids=(index, index_neigh), information=information, estimate=PoseSE2((pose[0], pose[1]), pose[2])
        )

    def _optimised_result_to_mst(self, result_params, tile_paths):
        """Converts the final globally optimised result that represents a grid of stitched images to a MST.

        :param result_params: Final optimised parameters.
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
        pose_root = result_params["img_0_x"], result_params["img_0_y"], result_params["img_0_angle"]
        root = TileNode(
            cfg=self.cfg,
            img=self.img_loader.load_img(tile_paths[position_root]),
            position=position_root,
            transformation=np.identity(3),
        )
        tile_nodes = [root]

        # Create the remaining nodes and connect them directly to the root node.
        for position, tile_path in np.ndenumerate(tile_paths):
            if position == position_root:
                continue

            # Load the current image.
            img = self.img_loader.load_img(tile_path)

            # Calculate translation and rotation matrices based on pose difference
            # to root.
            index = position[0] * grid_size[1] + position[1]
            pose_node = (
                result_params[f"img_{index}_x"],
                result_params[f"img_{index}_y"],
                result_params[f"img_{index}_angle"],
            )

            T = np.identity(3)
            R = np.identity(3)

            T[0, 2] = pose_node[0] - pose_root[0]
            T[1, 2] = pose_node[1] - pose_root[1]
            R[0, 0] = np.cos(pose_node[2] - pose_root[2])
            R[0, 1] = -np.sin(pose_node[2] - pose_root[2])
            R[1, 0] = -R[0, 1]
            R[1, 1] = R[0, 0]

            # Add the neighbor node to the MST.
            node = TileNode(cfg=self.cfg, img=img, parent=root, position=position, transformation=(T @ R))
            tile_nodes.append(node)

        return tile_nodes

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

        # Create the remaining nodes and connect them directly to the root node.
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
            node = TileNode(cfg=self.cfg, img=img, parent=root, position=position, transformation=(T @ R))
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
        for row, column in np.ndindex(grid_size):
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
        _, inliers_map = ransac(
            (matches, matches_neigh),
            transform_type,
            min_samples=min_samples,
            residual_threshold=self.cfg.STITCHER.RANSAC_THRESHOLD,
            max_trials=self.cfg.STITCHER.RANSAC_TRIALS,
            rng=self.cfg.STITCHER.RANSAC_SEED,
        )

        # Select matches to plot.
        inliers = matches[inliers_map]
        inliers_neigh = matches_neigh[inliers_map]
        match_samples = random.sample(range(0, len(inliers)), int(len(inliers) * fraction))

        # Prepare color coding.
        if color_conf:
            color = cm.jet(conf[match_samples])
        else:
            color = np.zeros((len(match_samples), 3))
            color[:, 1] = 1

        # Make the matching figure.
        fig_path = os.path.join(
            self.cfg.STITCHER.OUTPUT_PATH,
            "matches",
            f"{prefix}{position}_{position_neigh}_{len(inliers)}_{self.cfg.STITCHER.MATCHING_METHOD}_"
            f"{self.cfg.STITCHER.CONSTRUCTION_METHOD}.png",
        )
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        make_matching_figure(
            img, img_neigh, inliers[match_samples], inliers_neigh[match_samples], color, dpi=200, path=fig_path
        )

    def _save_pairwise_image(
        self,
        img,
        img_neigh,
        matches,
        matches_neigh,
        position,
        position_neigh,
        prefix="",
    ):
        """Stitches two neighboring tiles and saves the result.

        :param img: Current image tile.
        :param img_neigh: Neighboring image tile.
        :param matches: Matches of the current image tile.
        :param matches_neigh: Matches of the neighboring image tile.
        :param position: Position of the current image tile.
        :param position_neigh: Position of the neighboring image tile.
        :param prefix: Optional output prefix.
        """
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
        result, _ = ransac(
            (matches_neigh, matches),
            transform_type,
            min_samples=min_samples,
            residual_threshold=self.cfg.STITCHER.RANSAC_THRESHOLD,
            max_trials=self.cfg.STITCHER.RANSAC_TRIALS,
            rng=self.cfg.STITCHER.RANSAC_SEED,
        )

        # Create stitching trees representing the stitched images.
        root = TileNode(cfg=self.cfg, img=img, position=position, transformation=np.identity(3))
        node = TileNode(cfg=self.cfg, img=img_neigh, position=position_neigh, transformation=result.params)

        # Save the result.
        output_path = os.path.join(
            self.cfg.STITCHER.OUTPUT_PATH,
            "pairs",
            f"{prefix}{position}_{position_neigh}_{self.cfg.STITCHER.MATCHING_METHOD}_"
            f"{self.cfg.STITCHER.CONSTRUCTION_METHOD}.png",
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        stitched_image, _ = self.stitch_mst_nodes([root, node])
        stitched_image = cv2.resize(
            stitched_image,
            None,
            fx=self.cfg.STITCHER.SAVE_PAIRWISE_RESOLUTION_SCALE,
            fy=self.cfg.STITCHER.SAVE_PAIRWISE_RESOLUTION_SCALE,
        )
        cv2.imwrite(output_path, stitched_image)
