import pickle
import os, sys

from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from .base_dataset import BaseVolumeDataset


class KiTSVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        self.intensity_range = (-54, 247)
        self.target_spacing = (2, 1.5, 1.5)
        self.global_mean = 59.53867
        self.global_std = 55.457336
        self.spatial_index = [0, 1, 2]  # index used to convert to DHW
        self.do_dummy_2D = False
        self.target_class = 2


class LiTSVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        self.intensity_range = (-48, 163)
        self.target_spacing = (2.5, 1.5, 1.5)
        self.global_mean = 60.057533
        self.global_std = 40.198017
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = False
        self.target_class = 2


class PancreasVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        self.intensity_range = (-39, 204)
        self.target_spacing = (2.5, 1.5, 1.5)
        self.global_mean = 68.45214
        self.global_std = 63.422806
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 2


class ColonVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        self.intensity_range = (-57, 175)
        self.target_spacing = (2.5, 1.5, 1.5)
        self.global_mean = 65.175035
        self.global_std = 32.651197
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1


DATASET_DICT = {
    "kits": KiTSVolumeDataset,
    "lits": LiTSVolumeDataset,
    "msd": PancreasVolumeDataset,
    "colon": ColonVolumeDataset,
}


def load_data_volume(
    *,
    data,
    path_prefix,
    batch_size,
    data_dir=None,
    split="train",
    deterministic=False,
    augmentation=False,
    fold=0,
    rand_crop_spatial_size=(96, 96, 96),
    convert_to_sam=False,
    do_test_crop=True,
    do_nnunet_intensity_aug=False,
    num_worker=4,
):
    if not path_prefix:
        raise ValueError("unspecified data directory")
    if data_dir is None:
        data_dir = os.path.join(path_prefix, "split.pkl")

    with open(data_dir, "rb") as f:
        d = pickle.load(f)[fold][split]

    img_files = [os.path.join(path_prefix, d[i][0].strip("/")) for i in list(d.keys())]
    seg_files = [os.path.join(path_prefix, d[i][1].strip("/")) for i in list(d.keys())]

    dataset = DATASET_DICT[data](
        img_files,
        seg_files,
        batch_size=batch_size,
        split=split,
        augmentation=augmentation,
        rand_crop_spatial_size=rand_crop_spatial_size,
        convert_to_sam=convert_to_sam,
        do_test_crop=do_test_crop,
        do_nnunet_intensity_aug=do_nnunet_intensity_aug,
    )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, drop_last=True
        )
    return loader
