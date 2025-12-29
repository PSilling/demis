"""Evaluation module of the unlabeled grids of EM images.

Project: Deep Electron Microscopy Image Stitching (DEMIS)
Author: Petr Šilling
Year: 2024
"""

import os
from statistics import fmean
from timeit import default_timer

import cv2
import numpy as np
import torch
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.measure import ransac
from skimage.transform import AffineTransform, EuclideanTransform, ProjectiveTransform, SimilarityTransform
from torchvision.models.optical_flow import raft_small as raft

from src.dataset.dataset_loader import DatasetLoader
from src.pipeline.em424_stitcher import EM424Stitcher
from src.pipeline.image_loader import ImageLoader
from src.pipeline.tile_node import TileNode


class GridEvaluator:
    """Evaluation class for general grid datasets."""

    def __init__(
        self,
        cfg,
        transformations_path="",
        count=None,
    ) -> None:
        """GridEvaluator constructor.

        :param cfg: Grid evaluator configuration.
        :param transformations_path: Optional path to a directory with precomputed global tile transformations.
        :param count: Optional limit on the number of evaluated images.
        """
        self.cfg = cfg
        self.transformations_path = transformations_path
        self.img_loader = ImageLoader(cfg)
        self.results = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialise the stitcher.
        self.stitcher = EM424Stitcher(self.cfg, self.img_loader)

        # Initialise RAFT.
        self.raft = raft(pretrained=True, progress=False).to(self.device).eval()

        # Load dataset paths.
        loader = DatasetLoader(self.cfg.DATASET.PATH, self.cfg.DATASET.ROWS, self.cfg.DATASET.COLS)
        self.image_paths = loader.load_paths()

        # Limit the number of grid slices.
        if count is not None:
            self.image_paths = {k: self.image_paths[k] for k in list(self.image_paths)[:count]}

    def run(self):
        """Run evaluation of the dataset and print the results."""
        print("============================================================")
        print("Grid Dataset Evaluation")
        print("============================================================")
        print("Starting the evaluation...")
        print("============================================================")
        self._evaluate()

        # Print the results.
        print("Evaluation finished!")
        print("============================================================")
        print("Feature matching results:")
        print("============================================================")
        print(f"Mean time to stitch a single tile: {self.results['mean_seconds_per_image']:.4f} s")
        print(f"Mean time to compute pairwise feature matches: {self.results['mean_matching_seconds']:.4f} s")
        print(f"Mean number of matches: {self.results['mean_match_count']:d}")
        print(f"Mean number of inlier matches: {self.results['mean_inlier_count']:d}")

        print("============================================================")
        print("Image stitching results:")
        print("============================================================")
        print(f" EMSIQA-BASE: {fmean(self.results['EMSIQA-BASE']):.4f}")
        print(f"EMSIQA-RIDGE: {fmean(self.results['EMSIQA-RIDGE']):.4f}")
        print(f" EMSIQA-BOTH: {fmean(self.results['EMSIQA-BOTH']):.4f}")
        print(f" EMSIQA-FLOW: {fmean(self.results['EMSIQA-FLOW']):.4f}")

    def _get_neighboring_tiles(self, position, grid_size):
        """Retrieve neighbors of a given tile that should be processed next.
        Neighbors are selected in a top to bottom, left to right order.


        :param position: Target tile position.
        :param grid_size: Size of the whole grid.
        :return: List of unprocessed neighbors.
        """
        neighbors = []
        if position[0] < (grid_size[0] - 1):
            neighbors.append((position[0] + 1, position[1]))
        if position[1] < (grid_size[1] - 1):
            neighbors.append((position[0], position[1] + 1))
        return neighbors

    def _get_inliers(self, matches1, matches2):
        """Get inlier matches according to RANSAC.

        :param matches1: Matching keypoints from the source image tile.
        :param matches2: Matching keypoints from the target image tile.
        :return: Inliers matches (source first, target second).
        """
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
            return matches2[:1], matches1[:1]

        # Apply RANSAC to find inlier keypoints.
        _, inliers_map = ransac(
            (matches2, matches1),
            transform_type,
            min_samples=min_samples,
            residual_threshold=self.cfg.STITCHER.RANSAC_THRESHOLD,
            max_trials=self.cfg.STITCHER.RANSAC_TRIALS,
        )

        # Filter all outliers.
        return matches2[inliers_map], matches1[inliers_map]

    def _get_flow_errors(self, overlap_region1, overlap_region2, overlap_mask):
        """Calculates errors based on optical flow differences between two overlapping image regions.

        :param overlap_region1: Overlapping region of the first image.
        :param overlap_region2: Overlapping region of the second image.
        :param overlap_mask: Overlap mask of the image regions.
        :return: Calculated optical flow-based errors. Calculates four variants: original EMSIQA,
        EMSIQA with ridge detection instead of thresholding, a combined version of both, and
        a version without dice. All versions use RAFT instead of FlowNet.
        """
        # The images need to be in separate batches, with 3 channels and a resolution divisible by 8,
        # and scaled to [-1, 1] (the mask remains binary and has only 2 channels to match flow shape).
        shape = overlap_mask.shape
        if self.cfg.EVAL.RAFT_RESOLUTION_RATIO:
            shape = (shape[0] * self.cfg.EVAL.RAFT_RESOLUTION_RATIO, shape[1] * self.cfg.EVAL.RAFT_RESOLUTION_RATIO)
        region_size = (int(shape[1] // 8 * 8), int(shape[0] // 8 * 8))
        overlap_region1 = cv2.resize(overlap_region1, region_size)
        overlap_region2 = cv2.resize(overlap_region2, region_size)
        overlap_mask = cv2.resize(overlap_mask, region_size)

        region1_tensor = (torch.tensor(overlap_region1).to(self.device, dtype=torch.float32) / 255) * 2 - 1
        region2_tensor = (torch.tensor(overlap_region2).to(self.device, dtype=torch.float32) / 255) * 2 - 1

        region1_batch = torch.stack(3 * [region1_tensor]).unsqueeze(0)
        region2_batch = torch.stack(3 * [region2_tensor]).unsqueeze(0)

        # Predict optical flow.
        flow = self.raft(region1_batch, region2_batch)[-1]  # (N, 2, H, W), first channel is horizontal flow
        flow = flow[0].permute(1, 2, 0).cpu().detach().numpy()  # (H, W, 2)
        flow_transform = np.dstack(np.meshgrid(np.arange(overlap_mask.shape[1]), np.arange(overlap_mask.shape[0])))
        flow_transform = flow_transform.astype(np.float32) + flow

        # OTSU thresholding and median filtering.
        _, threshold1_otsu = cv2.threshold(overlap_region1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, threshold2_otsu = cv2.threshold(overlap_region2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if self.cfg.EVAL.INVERT_THRESHOLD:
            threshold1_otsu = cv2.bitwise_not(threshold1_otsu)
            threshold2_otsu = cv2.bitwise_not(threshold2_otsu)
        threshold1_otsu = cv2.medianBlur(cv2.bitwise_and(threshold1_otsu, overlap_mask), 5)
        threshold2_otsu = cv2.medianBlur(cv2.bitwise_and(threshold2_otsu, overlap_mask), 5)

        # Thresholding based on ridge detection.
        hessian1 = hessian_matrix(overlap_region1, sigma=1.5, use_gaussian_derivatives=False)
        hessian2 = hessian_matrix(overlap_region2, sigma=1.5, use_gaussian_derivatives=False)
        threshold1_ridges = hessian_matrix_eigvals(hessian1)[0]
        threshold2_ridges = hessian_matrix_eigvals(hessian2)[0]
        threshold1_ridges = cv2.normalize(threshold1_ridges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        threshold2_ridges = cv2.normalize(threshold2_ridges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, threshold1_ridges = cv2.threshold(threshold1_ridges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, threshold2_ridges = cv2.threshold(threshold2_ridges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold1_ridges = cv2.bitwise_and(threshold1_ridges, overlap_mask)
        threshold2_ridges = cv2.bitwise_and(threshold2_ridges, overlap_mask)

        threshold1_ridges = cv2.medianBlur(threshold1_ridges, 5)
        threshold2_ridges = cv2.medianBlur(threshold2_ridges, 5)

        threshold1_both = cv2.add(threshold1_ridges, threshold1_otsu)
        threshold2_both = cv2.add(threshold2_ridges, threshold2_otsu)

        emsiqa_results = {}
        threshold_inputs = [
            ("EMSIQA-BASE", threshold1_otsu, threshold2_otsu),
            ("EMSIQA-RIDGE", threshold1_ridges, threshold2_ridges),
            ("EMSIQA-BOTH", threshold1_both, threshold2_both),
        ]
        for name, threshold1, threshold2 in threshold_inputs:
            # Calculate flow metrics.
            flow_mask = np.dstack(2 * (threshold1.astype(np.float32) / 255,))
            masked_flow = flow * flow_mask
            flow_magnitude = np.mean(np.linalg.norm(masked_flow, axis=-1))

            # Calculate thresholding metrics.
            mask_transformed = cv2.remap(overlap_mask, flow_transform, None, cv2.INTER_NEAREST)
            threshold1_transformed = cv2.bitwise_and(threshold1, mask_transformed).astype(np.float32) / 255
            threshold2_transformed = cv2.remap(threshold2, flow_transform, None, cv2.INTER_NEAREST)
            threshold2_transformed = cv2.bitwise_and(threshold2_transformed, mask_transformed).astype(np.float32) / 255
            intersection = np.sum(threshold1_transformed * threshold2_transformed)
            dice = 2 * intersection / (np.sum(threshold1_transformed) + np.sum(threshold2_transformed))

            # Calculate EMSIQA.
            emsiqa_results[name] = flow_magnitude / dice

        # Add pure flow metrics.
        flow_mask = np.dstack(2 * (overlap_mask.astype(np.float32) / 255,))
        masked_flow = flow * flow_mask
        flow_magnitude = np.mean(np.linalg.norm(masked_flow, axis=-1))
        emsiqa_results["EMSIQA-FLOW"] = flow_magnitude
        return emsiqa_results

    def _align_images(self, img1, img2, center1, center2):
        """Aligns the sizes and centers of two images to the same position by
        zero padding.

        :param img1: First image.
        :param img2: Second image.
        :param center1: Center position of the first image.
        :param center2: Center position of the second image.
        :return: Aligned images and the new center position.
        """
        # Calculate paddings (left, top, right, bottom).
        padding1 = [0, 0, 0, 0]
        padding2 = [0, 0, 0, 0]
        center_diff = (round(abs(center1[0] - center2[0])), round(abs(center1[1] - center2[1])))
        if center1[0] < center2[0]:
            padding1[0] = center_diff[0]
        else:
            padding2[0] = center_diff[0]

        if center1[1] < center2[1]:
            padding1[1] = center_diff[1]
        else:
            padding2[1] = center_diff[1]

        size1 = (img1.shape[1] + padding1[0], img1.shape[0] + padding1[1])
        size2 = (img2.shape[1] + padding2[0], img2.shape[0] + padding2[1])
        size_diff = (abs(size1[0] - size2[0]), abs(size1[1] - size2[1]))
        if size1[0] < size2[0]:
            padding1[2] = size_diff[0]
        else:
            padding2[2] = size_diff[0]

        if size1[1] < size2[1]:
            padding1[3] = size_diff[1]
        else:
            padding2[3] = size_diff[1]

        # Pad the images with zeros.
        pad_value = [0, 0, 0] if img1.ndim > 2 else 0
        img1 = cv2.copyMakeBorder(
            img1, padding1[1], padding1[3], padding1[0], padding1[2], cv2.BORDER_CONSTANT, value=pad_value
        )
        img2 = cv2.copyMakeBorder(
            img2, padding2[1], padding2[3], padding2[0], padding2[2], cv2.BORDER_CONSTANT, value=pad_value
        )

        # Calculate the new center position.
        new_center = (max(center1[0], center2[0]), max(center1[1], center2[1]))

        return img1, img2, new_center

    def _evaluate(self):
        """Run the evaluation."""
        self.results = {
            "seconds_per_image": 0,
            "matching_seconds": 0,
            "match_count": 0,
            "inlier_count": 0,
            "mean_seconds_per_image": 0,
            "mean_matching_seconds": 0,
            "mean_match_count": 0,
            "mean_inlier_count": 0,
            "EMSIQA-BASE": [],
            "EMSIQA-RIDGE": [],
            "EMSIQA-BOTH": [],
            "EMSIQA-FLOW": [],
        }

        image_count = 0
        pair_count = 0
        for i, (path_key, tile_paths) in enumerate(self.image_paths.items()):
            grid_index, slice_index = path_key.split("_")
            print(f"[{i + 1}/{len(self.image_paths)}] Processing g{grid_index}_{slice_index}...")

            if self.transformations_path:
                # Load precomputed tile transformations.
                json_path = os.path.join(
                    self.transformations_path, f"g{int(grid_index):05d}_s{int(slice_index):05d}.json"
                )
                tile_transformations = self.stitcher.load_transformations(json_path)
            else:
                # Get globally optimised tile transformations.
                start_time = default_timer()
                tile_transformations = self.stitcher.stitch_grid(
                    tile_paths, transformations_only=True, plot_prefix=f"g{grid_index}_s{slice_index}_"
                )
                end_time = default_timer()
                self.results["seconds_per_image"] += end_time - start_time

            # Go over all possible pairs.
            image_count += len(tile_transformations)
            for position, tile_path in np.ndenumerate(tile_paths):
                neighbors = self._get_neighboring_tiles(position, tile_paths.shape)
                if not neighbors:
                    continue
                pair_count += len(neighbors)

                # Load the current image and process its neighbors.
                img = self.img_loader.load_img(tile_path)
                transformation = tile_transformations[position]
                for position_neigh in neighbors:
                    neighbor_path = tile_paths[position_neigh]
                    img_neigh = self.img_loader.load_img(neighbor_path)

                    # Calculate the relative transformation from global transformations.
                    transformation_neigh = tile_transformations[position_neigh]
                    transformation_relative = np.linalg.inv(transformation) @ transformation_neigh

                    # Compute keypoint matches.
                    start_time = default_timer()
                    matches, matches_neigh, _ = self.stitcher.compute_neighbor_matches(
                        position=position, position_neigh=position_neigh, img=img, img_neigh=img_neigh
                    )
                    end_time = default_timer()

                    # Get inlier matches.
                    inliers, _ = self._get_inliers(matches1=matches_neigh, matches2=matches)

                    self.results["matching_seconds"] += end_time - start_time
                    self.results["match_count"] += matches.shape[0]
                    self.results["inlier_count"] += inliers.shape[0]

                    # Create stitching trees representing the stitched images.
                    root = TileNode(cfg=self.cfg, img=img, position=position, transformation=np.identity(3))
                    node = TileNode(
                        cfg=self.cfg, img=img_neigh, position=position_neigh, transformation=transformation_relative
                    )

                    # Evaluate stitching results using optical flow.
                    warped_images, warped_masks, _ = self.stitcher.stitch_mst_nodes(
                        [root, node],
                        separate=True,
                        masks=True,
                        no_color=True,
                    )
                    overlap_regions, overlap_mask = self.stitcher.get_overlap_region(
                        warped_images, warped_masks, cropped=True
                    )
                    flow_errors = self._get_flow_errors(
                        overlap_regions[0],
                        overlap_regions[1],
                        overlap_mask,
                    )
                    self.results["EMSIQA-BASE"].append(flow_errors["EMSIQA-BASE"])
                    self.results["EMSIQA-RIDGE"].append(flow_errors["EMSIQA-RIDGE"])
                    self.results["EMSIQA-BOTH"].append(flow_errors["EMSIQA-BOTH"])
                    self.results["EMSIQA-FLOW"].append(flow_errors["EMSIQA-FLOW"])

        # Calculate mean statistics.
        self.results["mean_seconds_per_image"] = self.results["seconds_per_image"] / image_count
        self.results["mean_matching_seconds"] = self.results["matching_seconds"] / pair_count
        self.results["mean_match_count"] = round(self.results["match_count"] / pair_count)
        self.results["mean_inlier_count"] = round(self.results["inlier_count"] / pair_count)
