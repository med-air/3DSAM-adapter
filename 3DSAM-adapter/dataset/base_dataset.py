import pickle
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Union
from monai.config import IndexSelection, KeysCollection, SequenceStr
from monai.transforms import (
    Compose,
    AddChanneld,
    RandCropByPosNegLabeld,
    CropForegroundd,
    SpatialPadd,
    ScaleIntensityRanged,
    RandShiftIntensityd,
    RandFlipd,
    RandAffined,
    RandZoomd,
    RandRotated,
    RandRotate90d,
    RandGaussianNoised,
    RandGaussianSmoothd,
    NormalizeIntensityd,
    MapTransform,
    RandScaleIntensityd,
    RandSpatialCropd,
)
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F


class BinarizeLabeld(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            threshold: float = 0.5,
            allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.threshold = threshold

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            if not isinstance(d[key], torch.Tensor):
                d[key] = torch.as_tensor(d[key])

            dtype = d[key].dtype
            d[key] = (d[key] > self.threshold).to(dtype)
        return d


class BaseVolumeDataset(Dataset):
    def __init__(
            self,
            image_paths,
            label_meta,
            augmentation,
            split="train",
            rand_crop_spatial_size=(96, 96, 96),
            convert_to_sam=True,
            do_test_crop=True,
            do_val_crop=True,
            do_nnunet_intensity_aug=True,
    ):
        super().__init__()
        self.img_dict = image_paths
        self.label_dict = label_meta
        self.aug = augmentation
        self.split = split
        self.rand_crop_spatial_size = rand_crop_spatial_size
        self.convert_to_sam = convert_to_sam
        self.do_test_crop = do_test_crop
        self.do_nnunet_intensity_aug = do_nnunet_intensity_aug
        self.do_val_crop = do_val_crop
        self.intensity_range = (
            self.target_spacing
        ) = (
            self.global_mean
        ) = self.global_std = self.spatial_index = self.do_dummy_2D = self.target_class = None

        self._set_dataset_stat()
        self.transforms = self.get_transforms()

    def _set_dataset_stat(self):
        pass

    def __len__(self):
        return len(self.img_dict)

    def __getitem__(self, idx):
        img_path = self.img_dict[idx]
        label_path = self.label_dict[idx]

        img_vol = nib.load(img_path)
        img = img_vol.get_fdata().astype(np.float32).transpose(self.spatial_index)
        img_spacing = tuple(np.array(img_vol.header.get_zooms())[self.spatial_index])

        seg_vol = nib.load(label_path)
        seg = seg_vol.get_fdata().astype(np.float32).transpose(self.spatial_index)

        img[np.isnan(img)] = 0
        seg[np.isnan(seg)] = 0

        seg = (seg == self.target_class).astype(np.float32)
        if (np.max(img_spacing) / np.min(img_spacing) > 8) or (
                np.max(self.target_spacing / np.min(self.target_spacing) > 8)
        ):
            # resize 2D
            img_tensor = F.interpolate(
                input=torch.tensor(img[:, None, :, :]),
                scale_factor=tuple([img_spacing[i] / self.target_spacing[i] for i in range(1, 3)]),
                mode="bilinear",
            )

            if self.split != "test":
                seg_tensor = F.interpolate(
                    input=torch.tensor(seg[:, None, :, :]),
                    scale_factor=tuple(
                        [img_spacing[i] / self.target_spacing[i] for i in range(1, 3)]
                    ),
                    mode="bilinear",
                )
            img = (
                F.interpolate(
                    input=img_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous(),
                    scale_factor=(img_spacing[0] / self.target_spacing[0], 1, 1),
                    mode="trilinear",
                )
                .squeeze(0)
                .numpy()
            )

            if self.split != "test":
                seg = (
                    F.interpolate(
                        input=seg_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous(),
                        scale_factor=(img_spacing[0] / self.target_spacing[0], 1, 1),
                        mode="trilinear",
                    )
                    .squeeze(0)
                    .numpy()
                )
        else:
            img = (
                F.interpolate(
                    input=torch.tensor(img[None, None, :, :, :]),
                    scale_factor=tuple(
                        [img_spacing[i] / self.target_spacing[i] for i in range(3)]
                    ),
                    mode="trilinear",
                )
                .squeeze(0)
                .numpy()
            )
            if self.split != "test":
                seg = (
                    F.interpolate(
                        input=torch.tensor(seg[None, None, :, :, :]),
                        scale_factor=tuple(
                            [img_spacing[i] / self.target_spacing[i] for i in range(3)]
                        ),
                        mode="trilinear",
                    )
                    .squeeze(0)
                    .numpy()
                )

        if (self.aug and self.split == "train") or ((self.do_val_crop  and self.split=='val')):
            trans_dict = self.transforms({"image": img, "label": seg})[0]
            img_aug, seg_aug = trans_dict["image"], trans_dict["label"]
        else:
            trans_dict = self.transforms({"image": img, "label": seg})
            img_aug, seg_aug = trans_dict["image"], trans_dict["label"]
        seg_aug = seg_aug.squeeze()

        img_aug = img_aug.repeat(3, 1, 1, 1)

        return img_aug, seg_aug, np.array(img_vol.header.get_zooms())[self.spatial_index]

    def get_transforms(self):
        transforms = [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=self.intensity_range[0],
                a_max=self.intensity_range[1],
                b_min=self.intensity_range[0],
                b_max=self.intensity_range[1],
                clip=True,
            ),
        ]

        if self.split == "train":
            transforms.extend(
                [
                    RandShiftIntensityd(
                        keys=["image"],
                        offsets=20,
                        prob=0.5,
                    ),
                    CropForegroundd(
                        keys=["image", "label"],
                        source_key="image",
                        select_fn=lambda x: x > self.intensity_range[0],
                    ),
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.global_mean,
                        divisor=self.global_std,
                    ),
                ]
            )

            if self.do_dummy_2D:
                transforms.extend(
                    [
                       RandRotated(
                            keys=["image", "label"],
                            prob=0.3,
                            range_x=30 / 180 * np.pi,
                            keep_size=False,
                                ),
                        RandZoomd(
                            keys=["image", "label"],
                            prob=0.3,
                            min_zoom=[1, 0.9, 0.9],
                            max_zoom=[1, 1.1, 1.1],
                            mode=["trilinear", "trilinear"],
                        ),
                    ]
                )
            else:
                transforms.extend(
                    [
                        # RandRotated(
                        #     keys=["image", "label"],
                        #     prob=0.3,
                        #     range_x=30 / 180 * np.pi,
                        #     range_y=30 / 180 * np.pi,
                        #     range_z=30 / 180 * np.pi,
                        #     keep_size=False,
                        # ),
                        RandZoomd(
                            keys=["image", "label"],
                            prob=0.8,
                            min_zoom=0.85,
                            max_zoom=1.25,
                            mode=["trilinear", "trilinear"],
                        ),
                    ]
                )

            transforms.extend(
                [
                    BinarizeLabeld(keys=["label"]),
                    SpatialPadd(
                        keys=["image", "label"],
                        spatial_size=[round(i * 1.2) for i in self.rand_crop_spatial_size],
                    ),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        spatial_size=[round(i * 1.2) for i in self.rand_crop_spatial_size],
                        label_key="label",
                        pos=2,
                        neg=1,
                        num_samples=1,
                    ),
                    RandSpatialCropd(
                        keys=["image", "label"],
                        roi_size=self.rand_crop_spatial_size,
                        random_size=False,
                    ),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
                    # RandScaleIntensityd(keys="image", factors=0.1, prob=0.2),
                    # RandShiftIntensityd(
                    #     keys=["image"],
                    #     offsets=0.10,
                    #     prob=0.2,
                    # ),
                    # RandGaussianNoised(keys=["image"], prob=0.1),
                    # RandGaussianSmoothd(
                    #     keys=["image"],
                    #     prob=0.2,
                    #     sigma_x=(0.5, 1),
                    #     sigma_y=(0.5, 1),
                    #     sigma_z=(0.5, 1),
                    # ),
                    # AddChanneld(keys=["image", "label"]),
                    # RandShiftIntensityd(keys=["image"], offsets=10),
                    # RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),]
                ]
            )
        elif (not self.do_val_crop) and (self.split == "val"):
            transforms.extend(
                [
                    CropForegroundd(
                        keys=["image", "label"],
                        source_key="image",
                    ),
                    BinarizeLabeld(keys=["label"]),
                ]
            )
        elif  (self.do_val_crop)  and (self.split == "val"):
            transforms.extend(
                [
                    # CropForegroundd(
                    #     keys=["image", "label"],
                    #     source_key="image",
                    #     select_fn=lambda x: x > self.intensity_range[0],
                    # ),
                    SpatialPadd(
                        keys=["image", "label"],
                        spatial_size=[i for i in self.rand_crop_spatial_size],
                    ),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        spatial_size=self.rand_crop_spatial_size,
                        label_key="label",
                        pos=1,
                        neg=0,
                        num_samples=1,
                    ),
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.global_mean,
                        divisor=self.global_std,
                    ),
                    BinarizeLabeld(keys=["label"]),
                ]
            )
        elif self.split == "test":
            transforms.extend(
                [
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.global_mean,
                        divisor=self.global_std,
                    ),
                    BinarizeLabeld(keys=["label"]),
                ]
            )
        else:
            raise NotImplementedError

        transforms = Compose(transforms)

        return transforms