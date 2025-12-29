"""Evaluation module of the DEMIS tool.

Project: Deep Electron Microscopy Image Stitching (DEMIS)
Author: Petr Šilling
Year: 2023
"""

import os
import re
from statistics import fmean
from timeit import default_timer

import cv2
import numpy as np
import torch
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.measure import ransac
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from skimage.transform import AffineTransform, EuclideanTransform, ProjectiveTransform, SimilarityTransform
from torchvision.models.optical_flow import raft_small as raft

from LoFTR.src.utils.metrics import error_auc
from src.dataset.em424_loader import EM424Loader
from src.pipeline.em424_stitcher import EM424Stitcher
from src.pipeline.image_loader import ImageLoader
from src.pipeline.tile_node import TileNode


class DEMISEvaluator:
    """Evaluation class for the DEMIS pipeline. Expects to be used on the EM424 dataset."""

    def __init__(
        self,
        cfg,
        count=None,
    ) -> None:
        """DEMISEvaluator constructor.

        :param cfg: DEMIS evaluator configuration.
        :param count: Optional limit on the number of evaluated images.
        """
        self.cfg = cfg
        self.img_loader = ImageLoader(cfg)
        self.results = {}
        self.results_corner_error = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialise the stitcher.
        self.stitcher = EM424Stitcher(self.cfg, self.img_loader)

        # Initialise RAFT.
        self.raft = raft(pretrained=True, progress=False).to(self.device).eval()

        # Load EM424 labels and paths.
        loader = EM424Loader(self.cfg.DATASET.PATH)
        self.labels = loader.load_labels(self.cfg.EVAL.SPLIT_PATH)
        self.image_paths = loader.load_paths(self.labels)

        # Limit the number of labels.
        if count is not None:
            self.labels = self.labels[:count]

    def run(self):
        """Run the evaluation of the DEMIS pipeline."""
        print("============================================================")
        print("DEMIS Pipeline Evaluation")
        print("============================================================")
        print("Starting the evaluation...")
        print("============================================================")
        self._evaluate()

        # Print all results.
        print("Evaluation finished!")
        print("============================================================")
        print("Feature matching results:")
        print("============================================================")
        print(f"Mean time to stitch a single tile: {self.results['mean_seconds_per_image']:.4f} s")
        print(f"Mean time to compute pairwise feature matches: {self.results['mean_matching_seconds']:.4f} s")
        print(f"Mean number of matches: {self.results['mean_match_count']:d}")
        print(f"Mean number of inlier matches: {self.results['mean_inlier_count']:d}")
        print(f"Mean reprojection error of all matches: {self.results['mean_match_error']:.4f}")
        print(f"Mean reprojection error for inliers only:  {self.results['mean_inlier_error']:.4f}")
        for threshold in self.cfg.EVAL.ERROR_THRESHOLDS:
            print(f"Corner error AUC@{threshold:2d}px: {self.results_corner_error[str(threshold)] * 100:.4f}%")

        print("============================================================")
        print("Image stitching results:")
        print("============================================================")
        print(f"        RMSE: {fmean(self.results['RMSE']):.4f}")
        print(f"        PSNR: {fmean(self.results['PSNR']):.4f}")
        print(f"        SSIM: {fmean(self.results['SSIM']):.4f}")
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

    def _get_reprojection_error(self, matches1, matches2, transformation_gt):
        """Calculate the reprojection error between two image tiles.

        :param matches1: Matching keypoints from the source image tile.
        :param matches2: Matching keypoints from the target image tile.
        :param transformation_gt: Ground truth transformation to use for the error calculation.
        :return: Calculated reprojection error for each keypoint match.
        """
        # Convert source matches to target coordinate space.
        transformed_matches1 = np.pad(matches1, ((0, 0), (0, 1)), constant_values=1)
        transformed_matches1 = transformation_gt @ transformed_matches1.T
        transformed_matches1[0] /= transformed_matches1[2]
        transformed_matches1[1] /= transformed_matches1[2]
        transformed_matches1 = transformed_matches1[:2].T

        # Return the summed reprojection error and the number of matches used.
        return np.linalg.norm(transformed_matches1 - matches2, axis=1).sum()

    def _get_corner_error(self, img_shape, transformation_gt, transformation):
        """Calculate the corner error between two image tiles for all four corners.

        :param img_shape: Shape of the image tile.
        :param transformation_gt: Ground truth transformation to use for the error calculation.
        :param transformation: Transformation to evaluate against the ground truth one.
        :return: Calculated corner error for each tile corner.
        """
        # Get corner positions in homogenous coordinates.
        corners = np.array([[0, img_shape[1], img_shape[1], 0], [0, 0, img_shape[0], img_shape[0]], [1, 1, 1, 1]])

        # Concert the corners to the target coordinate space using each transformation.
        corners_gt = transformation_gt @ corners
        corners_gt[0] /= corners_gt[2]
        corners_gt[1] /= corners_gt[2]
        corners_gt = corners_gt[:2].T

        corners = transformation @ corners
        corners[0] /= corners[2]
        corners[1] /= corners[2]
        corners = corners[:2].T

        # Return the corner errors.
        return np.linalg.norm(corners - corners_gt, axis=1)

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
            "total_match_error": 0,
            "inlier_count": 0,
            "total_inlier_error": 0,
            "mean_seconds_per_image": 0,
            "mean_matching_seconds": 0,
            "mean_match_count": 0,
            "mean_match_error": 0,
            "mean_inlier_count": 0,
            "mean_inlier_error": 0,
            "RMSE": [],
            "PSNR": [],
            "SSIM": [],
            "EMSIQA-BASE": [],
            "EMSIQA-RIDGE": [],
            "EMSIQA-BOTH": [],
            "EMSIQA-FLOW": [],
        }

        psnr_infinity_counts = 0
        image_count = 0
        pair_count = 0
        corner_errors = []
        for i, labels in enumerate(self.labels):
            # Get tile paths.
            match = re.search(r"g(\d+)", os.path.basename(labels["path"]))
            if match is None:
                raise ValueError(f"Cannot parse labels file name: {labels['path']}.")
            grid_index = int(match.groups()[0])
            tile_labels = labels["tile_labels"]
            tile_paths = self.image_paths[f"{grid_index}_0"]

            print(f"[{i + 1}/{len(self.labels)}] Processing g{grid_index}...")

            # Get globally optimised tile transformations.
            start_time = default_timer()
            tile_transformations = self.stitcher.stitch_grid(
                tile_paths, transformations_only=True, plot_prefix=f"g{grid_index}_s00000_"
            )
            end_time = default_timer()
            self.results["seconds_per_image"] += end_time - start_time

            image_count += len(tile_transformations)

            # Go over all possible pairs.
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

                    # Calculate the ground truth homography.
                    position_index = position[0] * tile_paths.shape[1] + position[1]
                    position_index_neigh = position_neigh[0] * tile_paths.shape[1] + position_neigh[1]
                    transformation_gt = self.stitcher.get_transformation_between_tiles(
                        tile_labels1=tile_labels[position_index_neigh],
                        tile_labels2=tile_labels[position_index],
                        grid_labels=labels,
                    )

                    # Compute keypoint matches.
                    start_time = default_timer()
                    matches, matches_neigh, _ = self.stitcher.compute_neighbor_matches(
                        position=position, position_neigh=position_neigh, img=img, img_neigh=img_neigh
                    )
                    end_time = default_timer()

                    # Get inlier matches.
                    inliers, inliers_neigh = self._get_inliers(matches1=matches_neigh, matches2=matches)

                    self.results["matching_seconds"] += end_time - start_time
                    self.results["match_count"] += matches.shape[0]
                    self.results["inlier_count"] += inliers.shape[0]

                    # Calculate reprojection errors.
                    matches_error = self._get_reprojection_error(
                        matches1=matches_neigh, matches2=matches, transformation_gt=transformation_gt
                    )
                    inliers_error = self._get_reprojection_error(
                        matches1=inliers_neigh, matches2=inliers, transformation_gt=transformation_gt
                    )

                    self.results["total_match_error"] += matches_error
                    self.results["total_inlier_error"] += inliers_error

                    # Calculate corner errors.
                    matches_error = self._get_corner_error(
                        img_shape=img.shape,
                        transformation_gt=transformation_gt,
                        transformation=transformation_relative,
                    )
                    corner_errors += matches_error.tolist()

                    # Create stitching trees representing the stitched images.
                    root = TileNode(cfg=self.cfg, img=img, position=position, transformation=np.identity(3))
                    node_gt = TileNode(
                        cfg=self.cfg, img=img_neigh, position=position_neigh, transformation=transformation_gt
                    )
                    node = TileNode(
                        cfg=self.cfg, img=img_neigh, position=position_neigh, transformation=transformation_relative
                    )

                    # Stitch and evaluate the image pairs.
                    img_stitched, pos = self.stitcher.stitch_mst_nodes([root, node])
                    img_gt, pos_gt = self.stitcher.stitch_mst_nodes([root, node_gt])

                    # Align image centers and dimensions.
                    img_gt_aligned, img_aligned, _ = self._align_images(img_gt, img_stitched, pos_gt, pos)

                    # Evaluate the stitching.
                    psnr = peak_signal_noise_ratio(img_gt_aligned, img_aligned)
                    self.results["RMSE"].append(np.sqrt(mean_squared_error(img_gt_aligned, img_aligned)))
                    if np.isinf(psnr):
                        psnr_infinity_counts += 1
                    else:
                        self.results["PSNR"].append(psnr)
                    self.results["SSIM"].append(structural_similarity(img_gt_aligned, img_aligned))

                    # Evaluate stitching results using optical flow.
                    warped_images, warped_masks, _ = self.stitcher.stitch_mst_nodes(
                        [root, node], separate=True, masks=True
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

        # Calculate mean reprojection errors and counts.
        self.results["mean_seconds_per_image"] = self.results["seconds_per_image"] / image_count
        self.results["mean_matching_seconds"] = self.results["matching_seconds"] / pair_count
        self.results["mean_match_count"] = round(self.results["match_count"] / pair_count)
        self.results["mean_match_error"] = self.results["total_match_error"] / self.results["match_count"]
        self.results["mean_inlier_count"] = round(self.results["inlier_count"] / pair_count)
        self.results["mean_inlier_error"] = self.results["total_inlier_error"] / self.results["inlier_count"]

        # Replace infinities in PSNR by maximum detected PSNR from all other image pairs.
        psnr_max = np.max(self.results["PSNR"])
        self.results["PSNR"].extend(psnr_infinity_counts * [psnr_max])

        # Calculate AUC results (also add mean counts for consistency).
        self.results_corner_error = error_auc(corner_errors, self.cfg.EVAL.ERROR_THRESHOLDS, False)
