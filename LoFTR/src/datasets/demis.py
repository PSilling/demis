"""DEMIS dataset loader.

Project: Deep Electron Microscopy Image Stitching (DEMIS)
Author: Petr Šilling
Year: 2023
"""
import os.path as osp
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset

from src.utils.dataset import read_demis_gray, create_demis_depth


class DEMISDataset(Dataset):
    """Dataset loader for the DEMIS dataset."""

    def __init__(self, root_dir, npz_path, mode="train", img_resize=None,
                 df=None, img_padding=False, depth_padding=False, **kwargs):
        """
        Manage one labels file (npz_path) of DEMIS dataset. Data from labels files
        correspond to scenes of the other datasets. Inspired by the dataset class
        for MedaDepth dataset.

        Args:
            root_dir (str): DEMIS dataset root directory.
            npz_path (str): Path to image pair information (scene info).
            mode (str): Loading mode (train/val/test).
            img_resize (int, optional): The longer edge of resized images. None for no
                                        resize.
            df (int, optional): image size division factor. NOTE: this will change the
                                final image size after img_resize.
            img_padding (bool, optional): If set to True, zero-pad the image to
                                          squared size. This is useful during training.
            depth_padding (bool, optional): If set to True, zero-pad depthmap to shape
                                            (512, 512). This is necessary during
                                            training.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = npz_path.split('.')[0]

        # Load the scene info (representation of a single DEMIS grid).
        self.scene_info = np.load(npz_path, allow_pickle=True)
        self.pair_infos = deepcopy(self.scene_info["pair_infos"])
        self.image_paths = deepcopy(self.scene_info["image_paths"])
        self.poses = deepcopy(self.scene_info["poses"])

        # Prepare parameters for image resizing, padding and depthmap padding.
        if mode == "train":
            assert img_resize is not None and img_padding and depth_padding
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        self.depth_max_size = 512 if depth_padding else None

        # Training attributes.
        self.coarse_scale = getattr(kwargs, "coarse_scale", 0.125)

    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        idx0, idx1 = self.pair_infos[idx]

        # Read grayscale image with shape (1, h, w) and mask with shape (h, w).
        img_name0 = osp.join(self.root_dir, self.image_paths[idx0])
        img_name1 = osp.join(self.root_dir, self.image_paths[idx1])

        image0, _, scale0 = read_demis_gray(
            img_name0, self.img_resize, self.df, self.img_padding)
        image1, _, scale1 = read_demis_gray(
            img_name1, self.img_resize, self.df, self.img_padding)

        # Create a constant one depthmap with shape: (h, w), possibly zero padded
        # to (512, 512).
        if self.mode in ["train", "val"]:
            depth0 = create_demis_depth(image0.shape[1:], self.depth_max_size)
            depth1 = create_demis_depth(image1.shape[1:], self.depth_max_size)
        else:
            depth0 = torch.tensor([])
            depth1 = torch.tensor([])

        # Construct ideal intrinsics matrices.
        K_0 = torch.eye(3, dtype=torch.float)
        K_1 = torch.eye(3, dtype=torch.float)

        # Read image poses and compute relative poses. Since the poses are represented
        # directly via homographies (intrinsic matrices use identity), scaled versions
        # are calculated as well. For homography scaling, H' = S * H * S^-1 applies.
        H0 = self.poses[idx0]
        H1 = self.poses[idx1]

        S0 = torch.eye(4)
        S0[[0, 1], [0, 1]] = 1 / scale0
        S1 = torch.eye(4)
        S1[[0, 1], [0, 1]] = 1 / scale1

        H0 = np.matmul(S0, np.matmul(H0, np.linalg.inv(S0)))
        H1 = np.matmul(S1, np.matmul(H1, np.linalg.inv(S1)))
        scaledT_0to1 = np.matmul(np.linalg.inv(H1), H0).clone().detach().float()
        scaledT_1to0 = scaledT_0to1.inverse()

        data = {
            "image0": image0,  # (1, h, w)
            "depth0": depth0,  # (h, w)
            "image1": image1,
            "depth1": depth1,
            "T_0to1": scaledT_0to1,  # (4, 4)
            "T_1to0": scaledT_1to0,
            "K0": K_0,  # (3, 3)
            "K1": K_1,
            "dataset_name": "DEMIS",
            "scene_id": self.scene_id,
            "pair_id": idx,
            "pair_names": (self.image_paths[idx0],
                           self.image_paths[idx1]),
        }

        return data
