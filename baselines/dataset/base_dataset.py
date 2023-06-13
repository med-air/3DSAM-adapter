import pickle
import math
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Union
from monai.config import IndexSelection, KeysCollection, SequenceStr
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms import (
    Compose,
    RandCropByPosNegLabel,
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
    Decollated,
    Orientationd,
)
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from copy import deepcopy
from einops import rearrange


class StableRandCropByPosNegLabeld(RandCropByPosNegLabeld):
    def __init__(
        self,
        keys: KeysCollection,
        batch_size: int,
        label_key: str,
        spatial_size: Union[Sequence[int], int],
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image_key: Optional[str] = None,
        image_threshold: float = 0.0,
        fg_indices_key: Optional[str] = None,
        bg_indices_key: Optional[str] = None,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = "meta_dict",
        allow_smaller: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(
            keys,
            label_key,
            spatial_size,
            pos,
            neg,
            num_samples,
            image_key,
            image_threshold,
            fg_indices_key,
            bg_indices_key,
            allow_smaller,
            allow_missing_keys,
        )

        assert num_samples == 1

        self.batch_size = batch_size
        self.batch_pos_num = math.floor(batch_size * pos / (pos + neg))
        self.batch_neg_num = batch_size - self.batch_pos_num

        self.fg_cropper = RandCropByPosNegLabel(
            spatial_size=spatial_size,
            pos=1,
            neg=0,
            num_samples=1,
            image_threshold=image_threshold,
            allow_smaller=allow_smaller,
        )

        self.bg_cropper = RandCropByPosNegLabel(
            spatial_size=spatial_size,
            pos=0,
            neg=1,
            num_samples=1,
            image_threshold=image_threshold,
            allow_smaller=allow_smaller,
        )

        self.count = 0

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandCropByPosNegLabeld":
        super().set_random_state(seed, state)
        self.fg_cropper.set_random_state(seed, state)
        self.bg_cropper.set_random_state(seed, state)
        return self

    def randomize(
        self,
        label: torch.Tensor,
        fg_indices: Optional[NdarrayOrTensor] = None,
        bg_indices: Optional[NdarrayOrTensor] = None,
        image: Optional[torch.Tensor] = None,
    ) -> None:
        self.fg_cropper.randomize(
            label=label, fg_indices=fg_indices, bg_indices=bg_indices, image=image
        )
        self.bg_cropper.randomize(
            label=label, fg_indices=fg_indices, bg_indices=bg_indices, image=image
        )

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor]
    ) -> List[Dict[Hashable, torch.Tensor]]:
        d = dict(data)
        label = d[self.label_key]
        image = d[self.image_key] if self.image_key else None
        fg_indices = d.pop(self.fg_indices_key, None) if self.fg_indices_key is not None else None
        bg_indices = d.pop(self.bg_indices_key, None) if self.bg_indices_key is not None else None

        self.randomize(label, fg_indices, bg_indices, image)

        if (self.count % self.batch_size) < self.batch_pos_num:
            cropper = self.fg_cropper
        else:
            cropper = self.bg_cropper
        # initialize returned list with shallow copy to preserve key ordering
        ret: List = [dict(d) for _ in range(cropper.num_samples)]
        # deep copy all the unmodified data
        for i in range(cropper.num_samples):
            for key in set(d.keys()).difference(set(self.keys)):
                ret[i][key] = deepcopy(d[key])

        for key in self.key_iterator(d):
            for i, im in enumerate(cropper(d[key], label=label, randomize=False)):
                ret[i][key] = im

        self.count += 1

        return ret


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
        batch_size,
        split="train",
        rand_crop_spatial_size=(96, 96, 96),
        convert_to_sam=True,
        do_test_crop=True,
        do_nnunet_intensity_aug=True,
    ):
        super().__init__()
        self.img_dict = image_paths
        self.label_dict = label_meta
        self.aug = augmentation
        self.batch_size = batch_size
        self.split = split
        self.rand_crop_spatial_size = rand_crop_spatial_size
        self.convert_to_sam = convert_to_sam
        self.do_test_crop = do_test_crop
        self.do_nnunet_intensity_aug = do_nnunet_intensity_aug

        self.intensity_range = (
            self.target_spacing
        ) = (
            self.global_mean
        ) = (
            self.global_std
        ) = self.spatial_index = self.do_dummy_2D = self.target_class = self.num_classes = None

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
        seg_spacing = tuple(np.array(seg_vol.header.get_zooms())[self.spatial_index])

        img[np.isnan(img)] = 0
        seg[np.isnan(seg)] = 0

        if self.target_class is not None:
            seg = (seg == self.target_class).astype(np.float32)
            self.num_classes = 2

        assert self.num_classes is not None

        seg = rearrange(
            F.one_hot(torch.tensor(seg[:, :, :]).long(), num_classes=self.num_classes),
            "d h w c -> c d h w",
        ).float()

        if (np.max(img_spacing) / np.min(img_spacing) > 3) or (
            np.max(self.target_spacing / np.min(self.target_spacing) > 3)
        ):
            # resize 2D
            img_tensor = F.interpolate(
                input=torch.tensor(img[:, None, :, :]),
                scale_factor=tuple([img_spacing[i] / self.target_spacing[i] for i in range(1, 3)]),
                mode="bicubic",
            )

            if self.split != "test":
                seg_tensor = F.interpolate(
                    input=rearrange(seg, "c d h w -> d c h w"),
                    scale_factor=tuple(
                        [img_spacing[i] / self.target_spacing[i] for i in range(1, 3)]
                    ),
                    mode="bilinear",
                )

            # resize 3D
            img = (
                F.interpolate(
                    input=rearrange(img_tensor, f"d 1 h w -> 1 1 d h w"),
                    scale_factor=(img_spacing[0] / self.target_spacing[0], 1, 1),
                    mode="nearest",
                )
                .squeeze(0)
                .numpy()
            )

            if self.split != "test":
                seg = (
                    F.interpolate(
                        input=rearrange(seg_tensor, f"d c h w -> 1 c d h w"),
                        scale_factor=(img_spacing[0] / self.target_spacing[0], 1, 1),
                        mode="nearest",
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
                        input=seg.unsqueeze(0),
                        scale_factor=tuple(
                            [img_spacing[i] / self.target_spacing[i] for i in range(3)]
                        ),
                        mode="trilinear",
                    )
                    .squeeze(0)
                    .numpy()
                )

        if self.aug and self.split == "train":
            trans_dict = self.transforms({"image": img, "label": seg})[0]
            img_aug, seg_aug = trans_dict["image"], trans_dict["label"]

        else:
            trans_dict = self.transforms({"image": img, "label": seg})
            img_aug, seg_aug = trans_dict["image"], trans_dict["label"]

        seg_aug = seg_aug.squeeze().argmax(0)

        img_aug = img_aug.repeat(3, 1, 1, 1)

        if self.convert_to_sam:
            pass

        return img_aug, seg_aug, torch.tensor(img_spacing)

    def get_transforms(self):
        transforms = [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=self.intensity_range[0],
                a_max=self.intensity_range[1],
                b_min=0,
                b_max=1,
                clip=True,
            ),
        ]

        if self.split == "train" and self.aug:
            transforms.extend(
                [
                    CropForegroundd(
                        keys=["image", "label"],
                        source_key="image",
                        select_fn=lambda x: x > 0,
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
                        RandRotated(
                            keys=["image", "label"],
                            prob=0.3,
                            range_x=30 / 180 * np.pi,
                            range_y=30 / 180 * np.pi,
                            range_z=30 / 180 * np.pi,
                            keep_size=False,
                        ),
                        RandZoomd(
                            keys=["image", "label"],
                            prob=0.3,
                            min_zoom=0.9,
                            max_zoom=1.1,
                            mode=["trilinear", "trilinear"],
                        ),
                    ]
                )

            transforms.extend(
                [
                    BinarizeLabeld(keys=["label"]),
                    SpatialPadd(
                        keys=["image", "label"],
                        spatial_size=[int(i * 1.1) for i in self.rand_crop_spatial_size],
                    ),
                    StableRandCropByPosNegLabeld(
                        keys=["image", "label"],
                        batch_size=self.batch_size,
                        spatial_size=[int(i * 1.1) for i in self.rand_crop_spatial_size],
                        label_key="label",
                        pos=1,
                        neg=1,
                        num_samples=1,
                        image_key="image",
                        image_threshold=0,
                    ),
                    RandSpatialCropd(
                        keys=["image", "label"],
                        roi_size=self.rand_crop_spatial_size,
                        random_size=False,
                    ),
                    RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=[0]),
                    RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=[1]),
                    RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=[2]),
                    RandRotate90d(keys=["image", "label"], prob=0.1, spatial_axes=(1, 2)),
                    RandShiftIntensityd(
                        keys=["image"],
                        offsets=0.1,
                        prob=0.5,
                    ),
                    BinarizeLabeld(keys=["label"]),
                ]
            )

        elif self.split == "val":
            transforms.extend(
                [
                    CropForegroundd(
                        keys=["image", "label"],
                        source_key="image",
                    ),
                    BinarizeLabeld(keys=["label"]),
                ]
            )
        elif self.split == "test":
            transforms.extend(
                [
                    BinarizeLabeld(keys=["label"]),
                ]
            )
        else:
            raise NotImplementedError

        transforms = Compose(transforms)

        return transforms