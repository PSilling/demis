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
from skimage.transform import AffineTransform, ProjectiveTransform, SimilarityTransform
from torchvision.models.optical_flow import raft_small as raft
from torchvision.utils import flow_to_image

from LoFTR.src.utils.metrics import error_auc
from src.dataset.demis_loader import DemisLoader
from src.pipeline.demis_stitcher import DemisStitcher
from src.pipeline.image_loader import ImageLoader
from src.pipeline.tile_node import TileNode


class DEMISEvaluator:
    """Evaluation class for the DEMIS pipeline. Expects to be used on the DEMIS dataset."""

    def __init__(
        self,
        cfg,
        eval_matching=True,
        eval_homography=True,
        eval_pairs=True,
        eval_grid=True,
        count=None,
    ) -> None:
        """DEMISEvaluator constructor.

        :param cfg: DEMIS evaluator configuration.
        :param eval_matching: Whether to evaluate pairwise matching.
        :param eval_homography: Whether to evaluate pairwise homography estimation.
        :param eval_pairs: Whether to evaluate pairwise stitching output.
        :param eval_grid: Whether to evaluate the complete stitching output.
        :param count: Optional limit on the number of evaluated images.
        """
        self.cfg = cfg
        self.img_loader = ImageLoader(cfg)
        self.eval_matching = eval_matching
        self.eval_homography = eval_homography
        self.eval_pairs = eval_pairs
        self.eval_grid = eval_grid
        self.results_matching = {}
        self.results_homography = {}
        self.results_pairs = {}
        self.results_grid = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialise the stitcher.
        self.stitcher = DemisStitcher(self.cfg, self.img_loader)

        # Initialise RAFT.
        self.raft = raft(pretrained=True, progress=False).to(self.device).eval()

        # Load DEMIS labels and paths.
        loader = DemisLoader(self.cfg.DATASET.PATH)
        self.labels = loader.load_labels(self.cfg.EVAL.SPLIT_PATH)
        self.image_paths = loader.load_paths(self.labels)

        # Limit the number of labels.
        if count is not None:
            self.labels = self.labels[:count]

    def evaluate(self):
        """Run the evaluation of the DEMIS pipeline."""
        if not self.eval_matching and not self.eval_homography and not self.eval_pairs and not self.eval_grid:
            print("DEMIS Pipeline Evaluator: Nothing to evaluate!")

        # Run all chosen evaluations.
        print("============================================================")
        print("DEMIS Pipeline Evaluation")
        if self.eval_matching or self.eval_homography or self.eval_pairs:
            print("============================================================")
            print("Evaluating pairwise metrics...")
            print("============================================================")
            self._evaluate_pairwise()

        if self.eval_grid:
            print("============================================================")
            print("Evaluating the complete stitching output...")
            print("============================================================")
            self._evaluate_grid()

        # Print all results.
        print("Evaluation finished!")
        if self.eval_matching:
            print("============================================================")
            print("Pairwise matching results:")
            print("============================================================")
            print(f"Mean time to match a single pair of tiles: {self.results_matching['mean_seconds']:.4f} s")
            print(f"Mean reprojection error: {self.results_matching['mean_error']:.4f}")
            print(f"Mean inlier reprojection error:  {self.results_matching['mean_inlier_error']:.4f}")
            print(f"Mean matches count: {self.results_matching['mean_count']:d}")
            print(f"Mean inlier matches count: {self.results_matching['mean_inlier_count']:d}")

        if self.eval_homography:
            print("============================================================")
            print("Pairwise homography estimation results:")
            print("============================================================")
            if not self.eval_matching:
                # Add speed and match count information.
                print(f"Mean time to match a single pair of tiles: {self.results_matching['mean_seconds']:.4f} s")
                print(f"Mean matches count: {self.results_matching['mean_count']:d}")
                print(f"Mean inlier matches count: {self.results_matching['mean_inlier_count']:d}")
            for threshold in self.cfg.EVAL.ERROR_THRESHOLDS:
                print(f"Corner error AUC@{threshold:2d}px: {self.results_homography[str(threshold)] * 100:.4f}%")

        if self.eval_pairs:
            print("============================================================")
            print("Pairwise stitching output evaluation results:")
            print("============================================================")
            print(f"   RMSE: {fmean(self.results_pairs['RMSE']):.4f}")
            print(f"   PSNR: {fmean(self.results_pairs['PSNR']):.4f}")
            print(f"   SSIM: {fmean(self.results_pairs['SSIM']):.4f}")
            print(f"   FLOW: {fmean(self.results_pairs['FLOW']):.4f}")

        if self.eval_grid:
            print("============================================================")
            print("Complete grid stitching output evaluation results:")
            print("============================================================")
            print(f"   RMSE: {fmean(self.results_grid['RMSE']):.4f}")
            print(f"   PSNR: {fmean(self.results_grid['PSNR']):.4f}")
            print(f"   SSIM: {fmean(self.results_grid['SSIM']):.4f}")

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

    def _get_homography(self, img1, img2, position1, position2, stitcher):
        """Estimate a homography between two image tiles while respecting the limit of
        of total matches that can be used.

        :param img1: Source image tile.
        :param img2: Target image tile.
        :param position1: Position of the source tile in the grid.
        :param position2: Position of the target tile in the grid.
        :param stitcher: Stitcher to use for the estimation.
        :return: Estimated homography matrix, and the corresponding matches (target first).
        """
        # Compute keypoint matches.
        matches2, matches1, _ = stitcher.compute_neighbor_matches(
            position=position2, position_neigh=position1, img=img2, img_neigh=img1
        )

        # Estimate the evaluated homography from the matches.
        if self.cfg.STITCHER.TRANSFORM_TYPE == "affine":
            homography = np.identity(3)
            homography[:2, :], _ = cv2.estimateAffine2D(
                matches1, matches2, method=cv2.RANSAC, ransacReprojThreshold=self.cfg.STITCHER.RANSAC_THRESHOLD
            )
        elif self.cfg.STITCHER.TRANSFORM_TYPE == "partial-affine":
            homography = np.identity(3)
            homography[:2, :], _ = cv2.estimateAffinePartial2D(
                matches1, matches2, method=cv2.RANSAC, ransacReprojThreshold=self.cfg.STITCHER.RANSAC_THRESHOLD
            )
        elif self.cfg.STITCHER.TRANSFORM_TYPE == "perspective":
            homography, _ = cv2.findHomography(matches1, matches2, cv2.RANSAC, self.cfg.STITCHER.RANSAC_THRESHOLD)
        else:
            raise ValueError("Invalid transform type: {self.cfg.STITCHER.TRANSFORM_TYPE}")

        return homography, matches2, matches1

    def _get_inliers(self, matches1, matches2):
        """Get inlier matches according to RANSAC.

        :param matches1: Matching keypoints from the source image tile.
        :param matches2: Matching keypoints from the target image tile.
        :return: Inliers matches (source first, target second).
        """
        min_samples = 3
        if self.cfg.STITCHER.TRANSFORM_TYPE == "affine":
            transform_type = AffineTransform
        elif self.cfg.STITCHER.TRANSFORM_TYPE == "partial-affine":
            transform_type = SimilarityTransform
        elif self.cfg.STITCHER.TRANSFORM_TYPE == "perspective":
            transform_type = ProjectiveTransform
            min_samples = 4
        else:
            raise ValueError("Invalid transform type: {self.cfg.STITCHER.TRANSFORM_TYPE}")

        # Apply RANSAC to find inlier keypoints.
        _, inliers_map = ransac(
            (matches2, matches1),
            transform_type,
            min_samples=min_samples,
            residual_threshold=self.cfg.STITCHER.RANSAC_THRESHOLD,
            max_trials=100,
        )

        # Filter all outliers.
        return matches2[inliers_map], matches1[inliers_map]

    def _get_reprojection_error(self, matches1, matches2, homography_gt):
        """Calculate the reprojection error between two image tiles.

        :param matches1: Matching keypoints from the source image tile.
        :param matches2: Matching keypoints from the target image tile.
        :param homography_gt: Ground truth homography to use for the reprojection error calculation.
        :return: Calculated reprojection error for each keypoint match.
        """
        # Convert source matches to target coordinate space.
        transformed_matches1 = np.pad(matches1, ((0, 0), (0, 1)), constant_values=1)
        transformed_matches1 = homography_gt @ transformed_matches1.T
        transformed_matches1[0] /= transformed_matches1[2]
        transformed_matches1[1] /= transformed_matches1[2]
        transformed_matches1 = transformed_matches1[:2].T

        # Return the summed reprojection error and the number of matches used.
        return np.linalg.norm(transformed_matches1 - matches2, axis=1).sum()

    def _get_corner_error(self, img_shape, homography_gt, homography):
        """Calculate the corner error between two image tiles for all four corners.

        :param img_shape: Shape of the image tile.
        :param homography_gt: Ground truth homography to use for the corner error calculation.
        :param homography: Homography to evaluate against the ground truth one.
        :return: Calculated corner error for each tile corner.
        """
        # Get corner positions in homogenous coordinates.
        corners = np.array([[0, img_shape[1], img_shape[1], 0], [0, 0, img_shape[0], img_shape[0]], [1, 1, 1, 1]])

        # Concert the corners to the target coordinate space using each homography.
        corners_gt = homography_gt @ corners
        corners_gt[0] /= corners_gt[2]
        corners_gt[1] /= corners_gt[2]
        corners_gt = corners_gt[:2].T

        corners = homography @ corners
        corners[0] /= corners[2]
        corners[1] /= corners[2]
        corners = corners[:2].T

        # Return the corner errors.
        return np.linalg.norm(corners - corners_gt, axis=1)

    def _get_flow_error(self, overlap_region1, overlap_region2, overlap_mask):
        """Calculates error based on optical flow differences between two overlapping image regions.

        :param overlap_region1: Overlapping region of the first image.
        :param overlap_region2: Overlapping region of the second image.
        :param overlap_mask: Overlap mask of the image regions.
        :return: Calculated optical flow-based error.
        """
        # The images need to be in separate batches, with 3 channels and a resolution divisible by 8,
        # and scaled to [-1, 1] (the mask remains binary and has only 2 channels to match flow shape).
        region_size = (overlap_mask.shape[1] // 8 * 8, overlap_mask.shape[0] // 8 * 8)
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

        # # OTSU thresholding and median filtering.
        # _, threshold1_otsu = cv2.threshold(overlap_region1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # _, threshold2_otsu = cv2.threshold(overlap_region2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # threshold1_otsu = cv2.medianBlur(cv2.bitwise_and(cv2.bitwise_not(threshold1_otsu), overlap_mask), 5)
        # threshold2_otsu = cv2.medianBlur(cv2.bitwise_and(cv2.bitwise_not(threshold2_otsu), overlap_mask), 5)

        # # # Selecting threshold manually using edge detection.
        # region1_edges = cv2.Canny(overlap_region1, 100, 200, 5)
        # region2_edges = cv2.Canny(overlap_region2, 100, 200, 5)
        # region1_edges = cv2.bitwise_and(region1_edges, overlap_region1)
        # region2_edges = cv2.bitwise_and(region2_edges, overlap_region2)
        # threshold1_value = np.sum(region1_edges) / np.count_nonzero(region1_edges)
        # threshold2_value = np.sum(region2_edges) / np.count_nonzero(region2_edges)
        # _, threshold1_canny = cv2.threshold(overlap_region1, threshold1_value, 255, cv2.THRESH_BINARY)
        # _, threshold2_canny = cv2.threshold(overlap_region2, threshold2_value, 255, cv2.THRESH_BINARY)
        # threshold1_canny = cv2.medianBlur(cv2.bitwise_and(cv2.bitwise_not(threshold1_canny), overlap_mask), 5)
        # threshold2_canny = cv2.medianBlur(cv2.bitwise_and(cv2.bitwise_not(threshold2_canny), overlap_mask), 5)

        # Thresholding based on ridge detection.
        hessian1 = hessian_matrix(overlap_region1, sigma=2.5, use_gaussian_derivatives=False)
        hessian2 = hessian_matrix(overlap_region2, sigma=2.5, use_gaussian_derivatives=False)
        threshold1_ridges = hessian_matrix_eigvals(hessian1)[0]
        threshold2_ridges = hessian_matrix_eigvals(hessian2)[0]
        threshold1_ridges = cv2.normalize(threshold1_ridges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        threshold2_ridges = cv2.normalize(threshold2_ridges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, threshold1_ridges = cv2.threshold(threshold1_ridges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, threshold2_ridges = cv2.threshold(threshold2_ridges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold1_ridges = cv2.bitwise_and(threshold1_ridges, overlap_mask)
        threshold2_ridges = cv2.bitwise_and(threshold2_ridges, overlap_mask)

        threshold_inputs = [
            # (threshold1_otsu, threshold2_otsu),
            # (threshold1_canny, threshold2_canny),
            (threshold1_ridges, threshold2_ridges),
        ]
        for threshold1, threshold2 in threshold_inputs:
            # Calculate flow metrics.
            flow_mask = np.dstack(2 * (threshold1.astype(np.float32) / 255,))
            masked_flow = flow * flow_mask
            flow_magnitude = np.mean(np.linalg.norm(masked_flow, axis=-1))

            # Calculate thresholding metrics.
            flow_transform = np.dstack(np.meshgrid(np.arange(overlap_mask.shape[1]), np.arange(overlap_mask.shape[0])))
            flow_transform = flow_transform.astype(np.float32) + flow
            mask_transformed = cv2.remap(overlap_mask, flow_transform, None, cv2.INTER_NEAREST)
            threshold1_transformed = cv2.bitwise_and(threshold1, mask_transformed).astype(np.float32) / 255
            threshold2_transformed = cv2.remap(threshold2, flow_transform, None, cv2.INTER_NEAREST)
            threshold2_transformed = cv2.bitwise_and(threshold2_transformed, mask_transformed).astype(np.float32) / 255
            intersection = np.sum(threshold1_transformed * threshold2_transformed)
            dice = 2 * intersection / (np.sum(threshold1_transformed) + np.sum(threshold2_transformed))

            # Calculate EMSIQA.
            emsiqa = flow_magnitude / dice
            # print("EMSIQA:", emsiqa)

        # flow_imgs = flow_to_image(torch.tensor(masked_flow).permute(2, 0, 1).unsqueeze(0))
        # cv2.imwrite("output/overlap_0.png", overlap_region1)
        # cv2.imwrite("output/overlap_1.png", overlap_region2)
        # cv2.imwrite("output/threshold_0.png", threshold1_canny)
        # cv2.imwrite("output/threshold_1.png", threshold2_canny)
        # cv2.imwrite("output/threshold_otsu_0.png", threshold1_otsu)
        # cv2.imwrite("output/threshold_otsu_1.png", threshold2_otsu)
        # cv2.imwrite("output/threshold_0_transformed.png", threshold1_transformed * 255)
        # cv2.imwrite("output/threshold_1_transformed.png", threshold2_transformed * 255)
        # cv2.imwrite("output/threshold_ridges_0.png", threshold1_ridges)
        # cv2.imwrite("output/threshold_ridges_1.png", threshold2_ridges)
        # cv2.imwrite("output/edges_0.png", region1_edges)
        # cv2.imwrite("output/edges_1.png", region2_edges)
        # cv2.imwrite("output/mask.png", overlap_mask)
        # cv2.imwrite("output/mask_transformed.png", mask_transformed)
        # cv2.imwrite("output/flow.png", flow_imgs[0].permute(1, 2, 0).numpy())

        return emsiqa

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

    def _evaluate_pairwise(self):
        """Run the evaluation of pairwise matching and homography estimation (unless not selected)."""
        # Initialise result dictionaries.
        self.results_matching = {
            "total_seconds": 0,
            "sum": 0,
            "count": 0,
            "inlier_sum": 0,
            "inlier_count": 0,
            "mean_error": 0.0,
            "mean_count": 0,
            "mean_inlier_error": 0.0,
            "mean_inlier_count": 0,
        }
        self.results_homography = {}
        self.results_pairs = {
            "RMSE": [],
            "PSNR": [],
            "SSIM": [],
            "FLOW": [],
        }

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

            print(f"[{i + 1}/{len(self.labels)}] Processing pairs in grid starting with image {tile_paths[0, 0]}...")

            # Go over all possible pairs.
            for position, tile_path in np.ndenumerate(tile_paths):
                neighbors = self._get_neighboring_tiles(position, tile_paths.shape)
                if not neighbors:
                    continue
                pair_count += len(neighbors)

                # Load the current image and process its neighbors.
                img = self.img_loader.load_img(tile_path)
                for position_neigh in neighbors:
                    neighbor_path = tile_paths[position_neigh]
                    img_neigh = self.img_loader.load_img(neighbor_path)

                    # Calculate ground truth and evaluated homographies.
                    position_index = position[0] * tile_paths.shape[1] + position[1]
                    position_index_neigh = position_neigh[0] * tile_paths.shape[1] + position_neigh[1]

                    homography_gt = self.stitcher.get_transformation_between_tiles(
                        tile_labels1=tile_labels[position_index_neigh],
                        tile_labels2=tile_labels[position_index],
                        grid_labels=labels,
                    )

                    start_time = default_timer()
                    (homography, matches, matches_neigh) = self._get_homography(
                        img1=img_neigh,
                        img2=img,
                        position1=position_neigh,
                        position2=position,
                        stitcher=self.stitcher,
                    )
                    end_time = default_timer()

                    # Get inlier matches.
                    inliers, inliers_neigh = self._get_inliers(matches1=matches_neigh, matches2=matches)

                    if self.eval_matching:
                        # Calculate reprojection errors.
                        reprojection_error = self._get_reprojection_error(
                            matches1=matches_neigh, matches2=matches, homography_gt=homography_gt
                        )
                        inlier_error = self._get_reprojection_error(
                            matches1=inliers_neigh, matches2=inliers, homography_gt=homography_gt
                        )

                        # Increase the error sums and counts.
                        self.results_matching["sum"] += reprojection_error
                        self.results_matching["inlier_sum"] += inlier_error

                    # Always save time and match counts.
                    self.results_matching["total_seconds"] += end_time - start_time
                    self.results_matching["count"] += matches.shape[0]
                    self.results_matching["inlier_count"] += inliers.shape[0]

                    if self.eval_homography:
                        # Calculate corner errors.
                        reprojection_error = self._get_corner_error(
                            img_shape=img.shape, homography_gt=homography_gt, homography=homography
                        )

                        # Save the corner error data.
                        corner_errors += reprojection_error.tolist()

                    if self.eval_pairs:
                        # Create stitching trees representing the stitched images.
                        root = TileNode(cfg=self.cfg, img=img, position=position, transformation=np.identity(3))
                        node_gt = TileNode(
                            cfg=self.cfg, img=img_neigh, position=position_neigh, transformation=homography_gt
                        )
                        node = TileNode(cfg=self.cfg, img=img_neigh, position=position_neigh, transformation=homography)

                        # Stitch and evaluate the image pairs.
                        img_stitched, pos = self.stitcher.stitch_mst_nodes([root, node])
                        img_gt, pos_gt = self.stitcher.stitch_mst_nodes([root, node_gt])
                        self._compare_images(img_gt, img_stitched, pos_gt, pos, self.results_pairs)

                        # Evaluate stitching results using optical flow.
                        warped_images, warped_masks, _ = self.stitcher.stitch_mst_nodes(
                            [root, node], separate=True, masks=True
                        )
                        overlap_regions, overlap_mask = self.stitcher.get_overlap_region(
                            warped_images, warped_masks, cropped=True
                        )
                        self.results_pairs["FLOW"].append(
                            self._get_flow_error(overlap_regions[0], overlap_regions[1], overlap_mask)
                        )

        if self.eval_matching:
            # Calculate mean reprojection errors and counts.
            self.results_matching["mean_error"] = self.results_matching["sum"] / self.results_matching["count"]
            self.results_matching["mean_inlier_error"] = (
                self.results_matching["inlier_sum"] / self.results_matching["inlier_count"]
            )

        # Always calculate mean time and match counts.
        self.results_matching["mean_seconds"] = self.results_matching["total_seconds"] / pair_count
        self.results_matching["mean_count"] = round(self.results_matching["count"] / pair_count)
        self.results_matching["mean_inlier_count"] = round(self.results_matching["inlier_count"] / pair_count)

        if self.eval_homography:
            # Calculate AUC results (also add mean counts for consistency).
            self.results_homography = error_auc(corner_errors, self.cfg.EVAL.ERROR_THRESHOLDS, False)

    def _evaluate_grid(self):
        """Run the evaluation of the complete stitching output."""
        # Prepare a dictionary for the results.
        self.results_grid = {
            "RMSE": [],
            "PSNR": [],
            "SSIM": [],
        }

        for i, labels in enumerate(self.labels):
            # Get tile paths.
            match = re.search(r"g(\d+)", os.path.basename(labels["path"]))
            if match is None:
                raise ValueError(f"Cannot parse labels file name: {labels['path']}.")
            grid_index = int(match.groups()[0])
            tile_paths = self.image_paths[f"{grid_index}_0"]

            print(f"[{i + 1}/{len(self.labels)}] Processing grid starting with image {tile_paths[0, 0]}...")

            # Stitch the grid using ground truth labels and the selected method.
            img_gt, pos_gt = self.stitcher.stitch_demis_grid_mst(labels)
            img, pos = self.stitcher.stitch_grid(tile_paths)

            # Get the results.
            self._compare_images(img_gt, img, pos_gt, pos, self.results_grid)

    def _compare_images(self, img_gt, img, pos_gt, pos, result_dict):
        """Compares images to the ground truth image using metrics such as PSNR and SSIM.

        :param img_gt: Ground truth image.
        :param img: Image created using the selected method.
        :param pos_gt: Starting positions of the ground truth image.
        :param pos: Starting positions of the evaluated image.
        :param result_dict: Target dictionary to append results to.
        """
        # Align image centers and dimensions.
        img_gt, img, _ = self._align_images(img_gt, img, pos_gt, pos)

        # Evaluate the results.
        result_dict["RMSE"].append(np.sqrt(mean_squared_error(img_gt, img)))
        result_dict["PSNR"].append(peak_signal_noise_ratio(img_gt, img))
        result_dict["SSIM"].append(structural_similarity(img_gt, img))
