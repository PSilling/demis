"""Stitcher of pairs of overlapping image tiles.

Project: Deep Electron Microscopy Image Stitching (DEMIS)
Author: Petr Šilling
Year: 2023
"""

import cv2
import numpy as np
import torch


class TileStitcher:
    """Stitcher for neighbouring image tiles."""

    def __init__(self, cfg, img_processors):
        """TileStitcher constructor.

        :param cfg: DEMIS configuration.
        :param img_processors: Instances of image processing tools.
        """
        self.cfg = cfg
        self.img_processors = img_processors

    def resize_img(self, img):
        """Resize the given image.

        :param img: Image to resize.
        :return: Resized image.
        """
        ratio = self.cfg.STITCHER.RESOLUTION_RATIO
        divisibility = self.cfg.STITCHER.RESOLUTION_DIVISIBILITY

        # Respect maximum resolution.
        if round(img.shape[1] * ratio) > self.cfg.STITCHER.MAX_RESOLUTION:
            ratio = self.cfg.STITCHER.MAX_RESOLUTION / img.shape[1]
        if round(img.shape[0] * ratio) > self.cfg.STITCHER.MAX_RESOLUTION:
            ratio = self.cfg.STITCHER.MAX_RESOLUTION / img.shape[0]

        # Resize the image if necessary.
        if ratio != 1.0 or divisibility != 1:
            new_width = int(round(img.shape[1] * ratio) // divisibility * divisibility)
            new_height = int(round(img.shape[0] * ratio) // divisibility * divisibility)
            img = cv2.resize(img, (new_width, new_height))
        return img

    def crop_img(self, img, side="r"):
        """Crop the given image according to the expected tile overlap.

        :param img: Image to crop.
        :param side: From which side to crop (t/b/l/r for top/bottom/left/right).
        :return: Cropped image.
        """
        height, width = img.shape
        overlap = self.cfg.DATASET.TILE_OVERLAP
        if side == "t":
            return img[-round(height * overlap) :, :]
        if side == "b":
            return img[: round(height * overlap), :]
        if side == "l":
            return img[:, -round(width * overlap) :]
        if side == "r":
            return img[:, : round(width * overlap)]

        raise ValueError(f"Invalid crop side: {side}")

    def correct_keypoint_positions(self, kpts, img_full, img_crop, side="r"):
        """Correct positions of keypoints detected in a cropped image.

        :param kpts: Keypoints array.
        :param img_full: Complete uncropped image.
        :param img_crop: Cropped image.
        :param side: From which side cropping was performed (t/b/l/r for top/bottom/left/right).
        :return: Cropped image.
        """
        height_crop, width_crop = img_crop.shape
        height_full, width_full = img_full.shape

        # Correct keypoint resolution.
        if side == "t" or side == "b":
            resize_factor = width_full / width_crop
        elif side == "l" or side == "r":
            resize_factor = height_full / height_crop
        else:
            raise ValueError(f"Invalid crop side: {side}")
        kpts *= resize_factor

        # Correction keypoint positions for left and top tiles.
        if side == "t":
            kpts[:, 1] += height_full - (height_crop * resize_factor)
        elif side == "l":
            kpts[:, 0] += width_full - (width_crop * resize_factor)

        return kpts

    def compute_matches(self, img1_full, img2_full, horizontal=True):
        """
        Compute matches between a pair of images using the selected method.

        :param img1_full: Uncropped left/top image.
        :param img2_full: Uncropped light/bottom image.
        :param horizontal: Whether the images are horizontally aligned.
        :return: Detected matches and the corresponding confidence scores.
        """
        # Get cropped images based on expected overlap.
        if horizontal:
            img1_position = "l"
            img2_position = "r"
        else:
            img1_position = "t"
            img2_position = "b"

        img1 = self.crop_img(img1_full, img1_position)
        img2 = self.crop_img(img2_full, img2_position)

        # Resize the images. Note that LoFTR requires the final resolution to be
        # divisible by 8 and resolution in each dimension to be at most 2048.
        img1 = self.resize_img(img1)
        img2 = self.resize_img(img2)

        if self.cfg.STITCHER.MATCHING_METHOD == "loftr":
            # Create float batch tensors on the GPU.
            img1_batch = torch.from_numpy(img1)
            img1_batch = img1_batch.reshape(1, 1, *img1_batch.shape).cuda() / 255.0
            img2_batch = torch.from_numpy(img2)
            img2_batch = img2_batch.reshape(1, 1, *img2_batch.shape).cuda() / 255.0

            # Run inference with LoFTR to get match predictions.
            batch = {"image0": img1_batch, "image1": img2_batch}
            with torch.no_grad():
                self.img_processors["loftr"](batch)
                matches1 = batch["mkpts0_f"].cpu().numpy()
                matches2 = batch["mkpts1_f"].cpu().numpy()
                conf = batch["mconf"].cpu().numpy()
        elif self.cfg.STITCHER.MATCHING_METHOD in ("sift", "surf", "orb"):
            # Detect and describe SIFT/SURF/ORB features.
            kps1, desc1 = self.img_processors[self.cfg.STITCHER.MATCHING_METHOD].detectAndCompute(img1, None)
            kps2, desc2 = self.img_processors[self.cfg.STITCHER.MATCHING_METHOD].detectAndCompute(img2, None)

            # Match descriptors using the FLANN matcher.
            if self.cfg.STITCHER.MATCHING_METHOD == "orb":
                raw_matches = self.img_processors["bfmatcher"].knnMatch(desc1, desc2, k=2)
            else:
                raw_matches = self.img_processors["flann"].knnMatch(desc1, desc2, k=2)

            # Filter out bad matches using ratio test.
            conf = []
            matches = []
            for raw_match1, raw_match2 in raw_matches:
                distance_limit = raw_match2.distance * self.cfg.STITCHER.DISTANCE_RATIO
                if raw_match1.distance < distance_limit:
                    matches.append([raw_match1])
                    conf.append(1 - (raw_match1.distance / raw_match2.distance))

            # Get all matching points and confidence scores. Confidence scores are
            # calculated based on the distance between the best and second best matches.
            matches1 = np.float32([kps1[m.queryIdx].pt for [m] in matches])
            matches2 = np.float32([kps2[m.trainIdx].pt for [m] in matches])
            conf = np.float32(conf)
        else:
            raise ValueError("Invalid feature matching method: " f"{self.cfg.STITCHER.MATCHING_METHOD}")

        # Convert keypoint coordinates to full image space.
        matches1 = self.correct_keypoint_positions(matches1, img1_full, img1, img1_position)
        matches2 = self.correct_keypoint_positions(matches2, img2_full, img2, img2_position)
        return matches1, matches2, conf

    def stitch_tile(self, combined_image, tile_node, translation=None):
        """Stitch a single image tile in a grid. All transformations need to be
        calculated beforehand.

        :param combined_image: Combined target image for all image tiles in the grid.
        :param tile_node: Image tile to stitch.
        :param translation: Additional translation transformation matrix that should
                            be applied.
        :return: Updated combined image containing the stitched image tile.
        """
        img_tile = tile_node.img

        width = combined_image.shape[1]
        height = combined_image.shape[0]

        # The result is translated to the middle of a larger image to avoid cropping.
        M = tile_node.transformation
        if translation is not None:
            M = translation @ M

        # Warp the image tile.
        img_result = cv2.warpPerspective(img_tile, M, (width, height), flags=cv2.INTER_NEAREST)

        # Compose the final image.
        if self.cfg.STITCHER.COMPOSITING_METHOD == "overwrite":
            # Simply overwrite overlapping pixels by the original image.
            mask = combined_image > 0
            img_result[mask] = combined_image[mask]
        elif self.cfg.STITCHER.COMPOSITING_METHOD == "average":
            # Average overlapping pixel values.
            mask_both = (combined_image > 0) & (img_result > 0)
            mask_combined_only = (combined_image > 0) & (img_result == 0)

            img_result[mask_both] = 0.5 * combined_image[mask_both] + 0.5 * img_result[mask_both]
            img_result[mask_combined_only] = combined_image[mask_combined_only]
            img_result = np.rint(img_result).astype("uint8")
        elif self.cfg.STITCHER.COMPOSITING_METHOD == "adaptive":
            # Calculate adaptive weights for the top left region of the original tile.
            center = np.ceil((img_tile.shape[1] / 2, img_tile.shape[0] / 2)).astype("int")
            first_weights = (1 / center[0], 1 / center[1])
            last_weight = 0.5 / self.cfg.DATASET.TILE_OVERLAP

            weights_x = np.linspace(first_weights[0], last_weight, center[0]).clip(0, 1)
            weights_y = np.linspace(first_weights[1], last_weight, center[1]).clip(0, 1)

            weights_top_left = np.meshgrid(weights_x, weights_y)
            weights_top_left = weights_top_left[0] * weights_top_left[1]

            # Distribute the weights symmetrically to the whole image.
            weights = np.zeros((img_tile.shape[0], img_tile.shape[1]))
            weights[: weights_top_left.shape[0], : weights_top_left.shape[1]] = weights_top_left
            weights[: weights_top_left.shape[0], -weights_top_left.shape[1] :] = weights_top_left[:, ::-1]
            weights[-weights_top_left.shape[0] :] = weights[: weights_top_left.shape[0] :][::-1]

            # Warp the weights array the same way as the image tile.
            weights = cv2.warpPerspective(weights, M, (width, height), flags=cv2.INTER_NEAREST)

            # Correct dimensionality.
            if img_tile.ndim > 2:
                new_shape = (weights.shape[0], weights.shape[1], 3)
                weights = np.broadcast_to(weights[..., np.newaxis], new_shape)

            # Apply adaptive pixel weighting. Pixels closer to image tile center will
            # be weighted more heavily than those near the edges.
            mask_both = (combined_image > 0) & (img_result > 0)
            mask_combined_only = (combined_image > 0) & (img_result == 0)
            img_result[mask_both] = (1 - weights[mask_both]) * combined_image[mask_both] + weights[
                mask_both
            ] * img_result[mask_both]
            img_result[mask_combined_only] = combined_image[mask_combined_only]
            img_result = np.rint(img_result).astype("uint8")
        else:
            raise ValueError("Invalid compositing method: " f"{self.cfg.STITCHER.COMPOSITING_METHOD}")

        return img_result
