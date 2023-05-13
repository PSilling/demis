"""Evaluation module of the DEMIS tool.

Project: Deep Electron Microscopy Image Stitching (DEMIS)
Author: Petr Šilling
Year: 2023
"""
import cv2
import numpy as np
import os
import re
import torch
import warnings
from LoFTR.src.utils.metrics import error_auc
from piq import psnr, ssim, fsim, vsi, brisque
from skimage.measure import ransac
from skimage.metrics import mean_squared_error
from skimage.transform import SimilarityTransform, AffineTransform, ProjectiveTransform
from src.dataset.demis_loader import DemisLoader
from src.pipeline.demis_stitcher import DemisStitcher
from src.pipeline.image_loader import ImageLoader
from src.pipeline.tile_node import TileNode
from statistics import fmean
from timeit import default_timer


class DEMISEvaluator:
    """Evaluation class for the DEMIS pipeline. Expects to be used on the DEMIS
    dataset. The baseline for all evaluation metrics is SIFT."""

    def __init__(
        self,
        cfg,
        eval_matching=True,
        eval_homography=True,
        eval_pairs=True,
        eval_grid=True,
        count=None,
    ):
        """DEMISEvaluator constructor.

        :param cfg: DEMIS evaluator configuration.
        :param eval_matching: Whether to evaluate pairwise matching.
        :param eval_homography: Whether to evaluate pairwise homography estimation.
        :param eval_pairs: Whether to evaluate pairwise stitching output.
        :param eval_grid: Whether to evaluate the complete stitching output.
        :param count: Optional limit on the number of evaluated images.
        """
        self.img_loader = ImageLoader(cfg)
        self.eval_matching = eval_matching
        self.eval_homography = eval_homography
        self.eval_pairs = eval_pairs
        self.eval_grid = eval_grid
        self.results_matching = {}
        self.results_homography = {}
        self.results_pairs = {}
        self.results_grid = {}

        # Create matching configurations for stitching using LoFTR and SIFT.
        if cfg.STITCHER.MATCHING_METHOD == "sift":
            self.cfg_sift = cfg
            self.cfg_loftr = cfg.clone()
            self.cfg_loftr.STITCHER.MATCHING_METHOD = "loftr"
            self.cfg_loftr.freeze()
        else:
            self.cfg_sift = cfg.clone()
            self.cfg_sift.STITCHER.MATCHING_METHOD = "sift"
            self.cfg_sift.freeze()
            self.cfg_loftr = cfg

        # Initialise DEMIS stitchers.
        self.stitcher_sift = DemisStitcher(self.cfg_sift, self.img_loader)
        self.stitcher_loftr = DemisStitcher(self.cfg_loftr, self.img_loader)

        # Load DEMIS labels and paths.
        loader = DemisLoader(self.cfg_loftr.DATASET.PATH)
        self.labels = loader.load_labels(self.cfg_loftr.EVAL.SPLIT_PATH)
        self.image_paths = loader.load_paths(self.labels)

        # Limit the number of labels.
        if count is not None:
            self.labels = self.labels[:count]

        # Acknowledge the VSI RGB warning.
        warnings.filterwarnings(
            "ignore", message="The original VSI supports only RGB images"
        )

    def evaluate(self):
        """Run the evaluation of the DEMIS pipeline."""
        if (
            not self.eval_matching
            and not self.eval_homography
            and not self.eval_pairs
            and not self.eval_grid
        ):
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
            print(
                f"Mean time to match a single pair of tiles (SIFT):  "
                f"{self.results_matching['SIFT']['mean_seconds']:.4f} s"
            )
            print(
                f"Mean time to match a single pair of tiles (LoFTR): "
                f"{self.results_matching['LoFTR']['mean_seconds']:.4f} s"
            )
            print(
                f"Mean reprojection error (SIFT):  "
                f"{self.results_matching['SIFT']['mean_error']:.4f}"
            )
            print(
                f"Mean reprojection error (LoFTR): "
                f"{self.results_matching['LoFTR']['mean_error']:.4f}"
            )
            print(
                f"Mean inlier reprojection error (SIFT):  "
                f"{self.results_matching['SIFT']['mean_inlier_error']:.4f}"
            )
            print(
                f"Mean inlier reprojection error (LoFTR): "
                f"{self.results_matching['LoFTR']['mean_inlier_error']:.4f}"
            )
            print(
                f"Mean matches count (SIFT):  "
                f"{self.results_matching['SIFT']['mean_count']:d}"
            )
            print(
                f"Mean matches count (LoFTR): "
                f"{self.results_matching['LoFTR']['mean_count']:d}"
            )
            print(
                f"Mean inlier matches count (SIFT):  "
                f"{self.results_matching['SIFT']['mean_inlier_count']:d}"
            )
            print(
                f"Mean inlier matches count (LoFTR): "
                f"{self.results_matching['LoFTR']['mean_inlier_count']:d}"
            )

        if self.eval_homography:
            print("============================================================")
            print("Pairwise homography estimation results:")
            print("============================================================")
            if not self.eval_matching:
                # Add speed and match count information.
                print(
                    f"Mean time to match a single pair of tiles (SIFT):  "
                    f"{self.results_matching['SIFT']['mean_seconds']:.4f} s"
                )
                print(
                    f"Mean time to match a single pair of tiles (LoFTR): "
                    f"{self.results_matching['LoFTR']['mean_seconds']:.4f} s"
                )
                print(
                    f"Mean matches count (SIFT):  "
                    f"{self.results_matching['SIFT']['mean_count']:d}"
                )
                print(
                    f"Mean matches count (LoFTR): "
                    f"{self.results_matching['LoFTR']['mean_count']:d}"
                )
                print(
                    f"Mean inlier matches count (SIFT):  "
                    f"{self.results_matching['SIFT']['mean_inlier_count']:d}"
                )
                print(
                    f"Mean inlier matches count (LoFTR): "
                    f"{self.results_matching['LoFTR']['mean_inlier_count']:d}"
                )
            for threshold in self.cfg_loftr.EVAL.ERROR_THRESHOLDS:
                print(
                    f"Corner error AUC@{threshold:2d}px (SIFT):  "
                    f"{self.results_homography['SIFT'][str(threshold)] * 100:.4f}%"
                )
                print(
                    f"Corner error AUC@{threshold:2d}px (LoFTR): "
                    f"{self.results_homography['LoFTR'][str(threshold)] * 100:.4f}%"
                )

        if self.eval_pairs:
            print("============================================================")
            print("Pairwise stitching output evaluation results:")
            print("============================================================")
            print(f"   RMSE (SIFT):  {fmean(self.results_pairs['RMSE']['SIFT']):.4f}")
            print(f"   RMSE (LoFTR): {fmean(self.results_pairs['RMSE']['LoFTR']):.4f}")
            print(f"   PSNR (SIFT):  {fmean(self.results_pairs['PSNR']['SIFT']):.4f}")
            print(f"   PSNR (LoFTR): {fmean(self.results_pairs['PSNR']['LoFTR']):.4f}")
            print(f"   SSIM (SIFT):  {fmean(self.results_pairs['SSIM']['SIFT']):.4f}")
            print(f"   SSIM (LoFTR): {fmean(self.results_pairs['SSIM']['LoFTR']):.4f}")
            print(f"   FSIM (SIFT):  {fmean(self.results_pairs['FSIM']['SIFT']):.4f}")
            print(f"   FSIM (LoFTR): {fmean(self.results_pairs['FSIM']['LoFTR']):.4f}")
            print(f"    VSI (SIFT):  {fmean(self.results_pairs['VSI']['SIFT']):.4f}")
            print(f"    VSI (LoFTR): {fmean(self.results_pairs['VSI']['LoFTR']):.4f}")
            print(
                f"BRISQUE (SIFT):  {fmean(self.results_pairs['BRISQUE']['SIFT']):.4f}"
            )
            print(
                f"BRISQUE (LoFTR): {fmean(self.results_pairs['BRISQUE']['LoFTR']):.4f}"
            )

        if self.eval_grid:
            print("============================================================")
            print("Complete grid stitching output evaluation results:")
            print("============================================================")
            print(f"   RMSE (SIFT):  {fmean(self.results_grid['RMSE']['SIFT']):.4f}")
            print(f"   RMSE (LoFTR): {fmean(self.results_grid['RMSE']['LoFTR']):.4f}")
            print(f"   PSNR (SIFT):  {fmean(self.results_grid['PSNR']['SIFT']):.4f}")
            print(f"   PSNR (LoFTR): {fmean(self.results_grid['PSNR']['LoFTR']):.4f}")
            print(f"   SSIM (SIFT):  {fmean(self.results_grid['SSIM']['SIFT']):.4f}")
            print(f"   SSIM (LoFTR): {fmean(self.results_grid['SSIM']['LoFTR']):.4f}")
            print(f"   FSIM (SIFT):  {fmean(self.results_grid['FSIM']['SIFT']):.4f}")
            print(f"   FSIM (LoFTR): {fmean(self.results_grid['FSIM']['LoFTR']):.4f}")
            print(f"    VSI (SIFT):  {fmean(self.results_grid['VSI']['SIFT']):.4f}")
            print(f"    VSI (LoFTR): {fmean(self.results_grid['VSI']['LoFTR']):.4f}")
            print(f"BRISQUE (SIFT):  {fmean(self.results_grid['BRISQUE']['SIFT']):.4f}")
            print(
                f"BRISQUE (LoFTR): {fmean(self.results_grid['BRISQUE']['LoFTR']):.4f}"
            )

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
        :return: Estimated homography matrix, and the corresponding matches (target
                 first).
        """
        # Compute keypoint matches.
        matches2, matches1, conf = stitcher.compute_neighbor_matches(
            position=position2, position_neigh=position1, img=img2, img_neigh=img1
        )

        # Sort by confidence and choose the best matches.
        conf_indices = conf.argsort()[::-1]
        matches1 = matches1[conf_indices][: self.cfg_loftr.EVAL.MAX_MATCHES]
        matches2 = matches2[conf_indices][: self.cfg_loftr.EVAL.MAX_MATCHES]

        # Estimate the evaluated homography from the matches.
        if self.cfg_loftr.STITCHER.TRANSFORM_TYPE == "affine":
            homography = np.identity(3)
            homography[:2, :], _ = cv2.estimateAffine2D(
                matches1,
                matches2,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.cfg_loftr.STITCHER.RANSAC_THRESHOLD,
            )
        elif self.cfg_loftr.STITCHER.TRANSFORM_TYPE == "partial-affine":
            homography = np.identity(3)
            homography[:2, :], _ = cv2.estimateAffinePartial2D(
                matches1,
                matches2,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.cfg_loftr.STITCHER.RANSAC_THRESHOLD,
            )
        elif self.cfg_loftr.STITCHER.TRANSFORM_TYPE == "perspective":
            homography, _ = cv2.findHomography(
                matches1, matches2, cv2.RANSAC, self.cfg_loftr.STITCHER.RANSAC_THRESHOLD
            )
        else:
            raise ValueError(
                "Invalid transform type: " f"{self.cfg_loftr.STITCHER.TRANSFORM_TYPE}"
            )

        return homography, matches2, matches1

    def _get_inliers(self, matches1, matches2):
        """Get inlier matches according to RANSAC.

        :param matches1: Matching keypoints from the source image tile.
        :param matches2: Matching keypoints from the target image tile.
        :return: Inliers matches (source first, target second).
        """
        min_samples = 3
        if self.cfg_loftr.STITCHER.TRANSFORM_TYPE == "affine":
            transform_type = AffineTransform
        elif self.cfg_loftr.STITCHER.TRANSFORM_TYPE == "partial-affine":
            transform_type = SimilarityTransform
        elif self.cfg_loftr.STITCHER.TRANSFORM_TYPE == "perspective":
            transform_type = ProjectiveTransform
            min_samples = 4
        else:
            raise ValueError(
                "Invalid transform type: " f"{self.cfg_loftr.STITCHER.TRANSFORM_TYPE}"
            )

        # Apply RANSAC to find inlier keypoints.
        _, inliers_map = ransac(
            (matches2, matches1),
            transform_type,
            min_samples=min_samples,
            residual_threshold=self.cfg_loftr.STITCHER.RANSAC_THRESHOLD,
            max_trials=100,
        )

        # Filter all outliers.
        return matches2[inliers_map], matches1[inliers_map]

    def _get_reprojection_error(self, matches1, matches2, homography_gt):
        """Calculate the reprojection error between two image tiles.

        :param matches1: Matching keypoints from the source image tile.
        :param matches2: Matching keypoints from the target image tile.
        :param homography_gt: Ground truth homography to use for the reprojection
                              error calculation.
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
        :param homography_gt: Ground truth homography to use for the corner error
                              calculation.
        :param homography: Homography to evaluate against the ground truth one.
        :return: Calculated corner error for each tile corner.
        """
        # Get corner positions in homogenous coordinates.
        corners = np.array(
            [
                [0, img_shape[1], img_shape[1], 0],
                [0, 0, img_shape[0], img_shape[0]],
                [1, 1, 1, 1],
            ]
        )

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
        center_diff = (
            round(abs(center1[0] - center2[0])),
            round(abs(center1[1] - center2[1])),
        )
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
            img1,
            padding1[1],
            padding1[3],
            padding1[0],
            padding1[2],
            cv2.BORDER_CONSTANT,
            value=pad_value,
        )
        img2 = cv2.copyMakeBorder(
            img2,
            padding2[1],
            padding2[3],
            padding2[0],
            padding2[2],
            cv2.BORDER_CONSTANT,
            value=pad_value,
        )

        # Calculate the new center position.
        new_center = (max(center1[0], center2[0]), max(center1[1], center2[1]))

        return img1, img2, new_center

    def _evaluate_pairwise(self):
        """Run the evaluation of pairwise matching and homography estimation (unless
        not selected)."""
        # Initialise result dictionaries.
        self.results_matching = {
            "SIFT": {
                "total_seconds": 0,
                "sum": 0,
                "count": 0,
                "inlier_sum": 0,
                "inlier_count": 0,
                "mean_error": 0.0,
                "mean_count": 0,
                "mean_inlier_error": 0.0,
                "mean_inlier_count": 0,
            },
            "LoFTR": {
                "total_seconds": 0,
                "sum": 0,
                "count": 0,
                "inlier_sum": 0,
                "inlier_count": 0,
                "mean_error": 0.0,
                "mean_count": 0,
                "mean_inlier_error": 0.0,
                "mean_inlier_count": 0,
            },
        }
        self.results_homography = {"SIFT": {}, "LoFTR": {}}
        self.results_pairs = {
            "RMSE": {"SIFT": [], "LoFTR": []},
            "PSNR": {"SIFT": [], "LoFTR": []},
            "SSIM": {"SIFT": [], "LoFTR": []},
            "FSIM": {"SIFT": [], "LoFTR": []},
            "VSI": {"SIFT": [], "LoFTR": []},
            "BRISQUE": {"SIFT": [], "LoFTR": []},
        }

        pair_count = 0
        corner_errors_sift = []
        corner_errors_loftr = []
        for i, labels in enumerate(self.labels):
            # Get tile paths.
            match = re.search(r"g(\d+)", os.path.basename(labels["path"]))
            if match is None:
                raise ValueError(f"Cannot parse labels file name: {labels['path']}.")
            grid_index = int(match.groups()[0])
            tile_labels = labels["tile_labels"]
            tile_paths = self.image_paths[f"{grid_index}_0"]

            print(
                f"[{i + 1}/{len(self.labels)}] Processing pairs in grid starting "
                f"with image {tile_paths[0, 0]}..."
            )

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
                    position_index_neigh = (
                        position_neigh[0] * tile_paths.shape[1] + position_neigh[1]
                    )
                    homography_gt = (
                        self.stitcher_loftr.get_transformation_between_tiles(
                            tile_labels1=tile_labels[position_index_neigh],
                            tile_labels2=tile_labels[position_index],
                            grid_labels=labels,
                        )
                    )

                    start_time_sift = default_timer()
                    (
                        homography_sift,
                        matches_sift,
                        matches_sift_neigh,
                    ) = self._get_homography(
                        img1=img_neigh,
                        img2=img,
                        position1=position_neigh,
                        position2=position,
                        stitcher=self.stitcher_sift,
                    )
                    end_time_sift = default_timer()

                    start_time_loftr = default_timer()
                    (
                        homography_loftr,
                        matches_loftr,
                        matches_loftr_neigh,
                    ) = self._get_homography(
                        img1=img_neigh,
                        img2=img,
                        position1=position_neigh,
                        position2=position,
                        stitcher=self.stitcher_loftr,
                    )
                    end_time_loftr = default_timer()

                    # Get inlier matches.
                    inliers_sift, inliers_sift_neigh = self._get_inliers(
                        matches1=matches_sift_neigh, matches2=matches_sift
                    )
                    inliers_loftr, inliers_loftr_neigh = self._get_inliers(
                        matches1=matches_loftr_neigh, matches2=matches_loftr
                    )

                    if self.eval_matching:
                        # Calculate reprojection errors.
                        error_sift = self._get_reprojection_error(
                            matches1=matches_sift_neigh,
                            matches2=matches_sift,
                            homography_gt=homography_gt,
                        )
                        error_loftr = self._get_reprojection_error(
                            matches1=matches_loftr_neigh,
                            matches2=matches_loftr,
                            homography_gt=homography_gt,
                        )
                        inlier_error_sift = self._get_reprojection_error(
                            matches1=inliers_sift_neigh,
                            matches2=inliers_sift,
                            homography_gt=homography_gt,
                        )
                        inlier_error_loftr = self._get_reprojection_error(
                            matches1=inliers_loftr_neigh,
                            matches2=inliers_loftr,
                            homography_gt=homography_gt,
                        )

                        # Increase the error sums and counts.
                        self.results_matching["SIFT"]["sum"] += error_sift
                        self.results_matching["LoFTR"]["sum"] += error_loftr
                        self.results_matching["SIFT"]["inlier_sum"] += inlier_error_sift
                        self.results_matching["LoFTR"][
                            "inlier_sum"
                        ] += inlier_error_loftr

                    # Always save time and match counts.
                    self.results_matching["SIFT"]["total_seconds"] += (
                        end_time_sift - start_time_sift
                    )
                    self.results_matching["LoFTR"]["total_seconds"] += (
                        end_time_loftr - start_time_loftr
                    )
                    self.results_matching["SIFT"]["count"] += matches_sift.shape[0]
                    self.results_matching["LoFTR"]["count"] += matches_loftr.shape[0]
                    self.results_matching["SIFT"]["inlier_count"] += inliers_sift.shape[
                        0
                    ]
                    self.results_matching["LoFTR"][
                        "inlier_count"
                    ] += inliers_loftr.shape[0]

                    if self.eval_homography:
                        # Calculate corner errors.
                        error_sift = self._get_corner_error(
                            img_shape=img.shape,
                            homography_gt=homography_gt,
                            homography=homography_sift,
                        )
                        error_loftr = self._get_corner_error(
                            img_shape=img.shape,
                            homography_gt=homography_gt,
                            homography=homography_loftr,
                        )

                        # Save the corner error data.
                        corner_errors_sift += error_sift.tolist()
                        corner_errors_loftr += error_loftr.tolist()

                    if self.eval_pairs:
                        # Create stitching trees representing the stitched images.
                        root = TileNode(
                            cfg=self.cfg_loftr,
                            img=img,
                            position=position,
                            transformation=np.identity(3),
                        )
                        node_gt = TileNode(
                            cfg=self.cfg_loftr,
                            img=img_neigh,
                            position=position_neigh,
                            transformation=homography_gt,
                        )
                        node_sift = TileNode(
                            cfg=self.cfg_sift,
                            img=img_neigh,
                            position=position_neigh,
                            transformation=homography_sift,
                        )
                        node_loftr = TileNode(
                            cfg=self.cfg_loftr,
                            img=img_neigh,
                            position=position_neigh,
                            transformation=homography_loftr,
                        )

                        # Stitch and evaluate the image pairs.
                        img_gt, pos_gt = self.stitcher_loftr.stitch_mst_nodes(
                            [root, node_gt]
                        )
                        img_sift, pos_sift = self.stitcher_sift.stitch_mst_nodes(
                            [root, node_sift]
                        )
                        img_loftr, pos_loftr = self.stitcher_loftr.stitch_mst_nodes(
                            [root, node_loftr]
                        )
                        self._compare_images(
                            img_gt,
                            img_sift,
                            img_loftr,
                            pos_gt,
                            pos_sift,
                            pos_loftr,
                            self.results_pairs,
                        )

        if self.eval_matching:
            # Calculate mean reprojection errors and counts.
            self.results_matching["SIFT"]["mean_error"] = (
                self.results_matching["SIFT"]["sum"]
                / self.results_matching["SIFT"]["count"]
            )
            self.results_matching["LoFTR"]["mean_error"] = (
                self.results_matching["LoFTR"]["sum"]
                / self.results_matching["LoFTR"]["count"]
            )
            self.results_matching["SIFT"]["mean_inlier_error"] = (
                self.results_matching["SIFT"]["inlier_sum"]
                / self.results_matching["SIFT"]["inlier_count"]
            )
            self.results_matching["LoFTR"]["mean_inlier_error"] = (
                self.results_matching["LoFTR"]["inlier_sum"]
                / self.results_matching["LoFTR"]["inlier_count"]
            )

        # Always calculate mean time and match counts.
        self.results_matching["SIFT"]["mean_seconds"] = (
            self.results_matching["SIFT"]["total_seconds"] / pair_count
        )
        self.results_matching["LoFTR"]["mean_seconds"] = (
            self.results_matching["LoFTR"]["total_seconds"] / pair_count
        )
        self.results_matching["SIFT"]["mean_count"] = round(
            self.results_matching["SIFT"]["count"] / pair_count
        )
        self.results_matching["LoFTR"]["mean_count"] = round(
            self.results_matching["LoFTR"]["count"] / pair_count
        )
        self.results_matching["SIFT"]["mean_inlier_count"] = round(
            self.results_matching["SIFT"]["inlier_count"] / pair_count
        )
        self.results_matching["LoFTR"]["mean_inlier_count"] = round(
            self.results_matching["LoFTR"]["inlier_count"] / pair_count
        )

        if self.eval_homography:
            # Calculate AUC results (also add mean counts for consistency).
            self.results_homography["SIFT"] = error_auc(
                corner_errors_sift, self.cfg_loftr.EVAL.ERROR_THRESHOLDS, False
            )
            self.results_homography["LoFTR"] = error_auc(
                corner_errors_loftr, self.cfg_loftr.EVAL.ERROR_THRESHOLDS, False
            )

    def _evaluate_grid(self):
        """Run the evaluation of the complete stitching output."""
        # Prepare a dictionary for the results.
        self.results_grid = {
            "RMSE": {"SIFT": [], "LoFTR": []},
            "PSNR": {"SIFT": [], "LoFTR": []},
            "SSIM": {"SIFT": [], "LoFTR": []},
            "FSIM": {"SIFT": [], "LoFTR": []},
            "VSI": {"SIFT": [], "LoFTR": []},
            "BRISQUE": {"SIFT": [], "LoFTR": []},
        }

        for i, labels in enumerate(self.labels):
            # Get tile paths.
            match = re.search(r"g(\d+)", os.path.basename(labels["path"]))
            if match is None:
                raise ValueError(f"Cannot parse labels file name: {labels['path']}.")
            grid_index = int(match.groups()[0])
            tile_paths = self.image_paths[f"{grid_index}_0"]

            print(
                f"[{i + 1}/{len(self.labels)}] Processing grid starting with "
                f"image {tile_paths[0, 0]}..."
            )

            # Stitch the grid using ground truth labels, LoFTR, and SIFT.
            img_gt, pos_gt = self.stitcher_loftr.stitch_demis_grid_mst(labels)
            img_sift, pos_sift = self.stitcher_sift.stitch_grid(tile_paths)
            img_loftr, pos_loftr = self.stitcher_loftr.stitch_grid(tile_paths)

            # Get the results.
            # Note: Generally does not give results in line with pairwise stitching.
            # There might be issues with regards to grid center positioning etc.
            self._compare_images(
                img_gt,
                img_sift,
                img_loftr,
                pos_gt,
                pos_sift,
                pos_loftr,
                self.results_grid,
            )

    def _compare_images(
        self, img_gt, img_sift, img_loftr, pos_gt, pos_sift, pos_loftr, result_dict
    ):
        """Compares SIFT and LoFTR images to the ground truth image using metrics
        such as PSNR and SSIM.

        :param img_gt: Ground truth image.
        :param img_sift: Image created using SIFT.
        :param img_loftr: Image created using LoFTR.
        :param pos_gt: Starting positions of the ground truth image.
        :param pos_sift: Starting positions of the SIFT image.
        :param pos_loftr: Starting positions of the LoFTR image.
        :param result_dict: Target dictionary to append results to.
        """
        # Align image centers and dimensions.
        img_gt, img_sift, pos_gt_sift = self._align_images(
            img_gt, img_sift, pos_gt, pos_sift
        )
        img_gt, img_loftr, pos_gt_loftr = self._align_images(
            img_gt, img_loftr, pos_gt_sift, pos_loftr
        )
        img_sift, img_loftr, _ = self._align_images(
            img_sift, img_loftr, pos_gt_sift, pos_gt_loftr
        )

        # Get tensor representations of the results (necessary for PIQ metrics).
        if img_gt.ndim > 2:
            img_tensor_gt = torch.from_numpy(img_gt).permute(2, 0, 1)[np.newaxis, ...]
            img_tensor_sift = torch.from_numpy(img_sift).permute(2, 0, 1)[
                np.newaxis, ...
            ]
            img_tensor_loftr = torch.from_numpy(img_loftr).permute(2, 0, 1)[
                np.newaxis, ...
            ]
        else:
            img_tensor_gt = torch.from_numpy(img_gt)[np.newaxis, np.newaxis, ...]
            img_tensor_sift = torch.from_numpy(img_sift)[np.newaxis, np.newaxis, ...]
            img_tensor_loftr = torch.from_numpy(img_loftr)[np.newaxis, np.newaxis, ...]

        # Evaluate the results.
        result_dict["RMSE"]["SIFT"].append(
            np.sqrt(mean_squared_error(img_gt, img_sift))
        )
        result_dict["RMSE"]["LoFTR"].append(
            np.sqrt(mean_squared_error(img_gt, img_loftr))
        )
        result_dict["PSNR"]["SIFT"].append(
            min(psnr(img_tensor_gt, img_tensor_sift, data_range=255).item(), 80.0)
        )
        result_dict["PSNR"]["LoFTR"].append(
            min(psnr(img_tensor_gt, img_tensor_loftr, data_range=255).item(), 80.0)
        )
        result_dict["SSIM"]["SIFT"].append(
            ssim(
                img_tensor_gt, img_tensor_sift, data_range=255, downsample=False
            ).item()
        )
        result_dict["SSIM"]["LoFTR"].append(
            ssim(
                img_tensor_gt, img_tensor_loftr, data_range=255, downsample=False
            ).item()
        )
        result_dict["FSIM"]["SIFT"].append(
            fsim(
                img_tensor_gt,
                img_tensor_sift,
                data_range=255,
                chromatic=img_gt.ndim > 2,
            ).item()
        )
        result_dict["FSIM"]["LoFTR"].append(
            fsim(
                img_tensor_gt,
                img_tensor_loftr,
                data_range=255,
                chromatic=img_gt.ndim > 2,
            ).item()
        )
        result_dict["VSI"]["SIFT"].append(
            vsi(img_tensor_gt, img_tensor_sift, data_range=255).item()
        )
        result_dict["VSI"]["LoFTR"].append(
            vsi(img_tensor_gt, img_tensor_loftr, data_range=255).item()
        )
        result_dict["BRISQUE"]["SIFT"].append(
            brisque(img_tensor_sift, data_range=255).item()
        )
        result_dict["BRISQUE"]["LoFTR"].append(
            brisque(img_tensor_loftr, data_range=255).item()
        )
