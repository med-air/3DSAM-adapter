import os
from dataset.datasets import load_data_volume
import argparse

import torch
import numpy as np
import logging
from utils.model_util import get_model
import torch.nn.functional as F

from monai.losses import DiceCELoss, DiceLoss
from monai.inferers import sliding_window_inference

from utils.util import setup_logger
import surface_distance
from surface_distance import metrics


def set_default_arguments(args):
    args.rand_crop_size = (64, 160, 160)

    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default=None, type=str, choices=["kits", "pancreas", "lits", "colon"]
    )
    parser.add_argument("--data_prefix", default=None, type=str)
    parser.add_argument(
        "--snapshot_path",
        default="",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--method",
        default=None,
        type=str,
        choices=["swin_unetr", "unetr", "3d_uxnet", "nnformer", "unetr++", "transbts"],
    )
    parser.add_argument(
        "--data_prefix",
        default="",
        type=str,
    )
    parser.add_argument("--overlap", default=0.7, type=float)
    parser.add_argument(
        "--infer_mode", default="constant", type=str, choices=["constant", "gaussian"]
    )
    parser.add_argument("--save_pred", action="store_true")
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--num_worker", default=6, type=int)
    parser.add_argument("--tolerance", default=5, type=int)

    args = parser.parse_args()
    args = set_default_arguments(args)
    args.snapshot_path = os.path.join(args.snapshot_path, args.data, args.method)
    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)

    setup_logger(logger_name="test", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"test")
    logger.info(str(args))

    test_data = load_data_volume(
        data=args.data,
        batch_size=1,
        path_prefix=args.data_prefix,
        augmentation=False,
        split="test",
        convert_to_sam=False,
        num_worker=args.num_worker,
        deterministic=True,
    )

    seg_net = get_model(args).cuda()
    loss_val = DiceLoss(
        softmax=False, include_background=False, reduction="none", to_onehot_y=True
    )
    ckpt = torch.load(os.path.join(args.snapshot_path, "best.pth.tar"))

    seg_net.load_state_dict(ckpt["network_dict"])
    logger.info(f"Loading checkpoint done!")
    logger.info("Best validation Dice: {:.6f}".format(1 - ckpt["best_val_loss"]))

    logger.info(
        "#Param: {}".format(sum(p.numel() for p in seg_net.parameters() if p.requires_grad))
    )

    loss_summary = []
    nsd_list = []

    if args.save_pred:
        save_pred_path = os.path.join(args.snapshot_path, "predictions")
        if not os.path.exists(save_pred_path):
            os.makedirs(save_pred_path)

    with torch.no_grad():
        seg_net.eval()
        for idx, (img, seg, spacing) in enumerate(test_data):
            img = img.cuda().float()
            img = img[:, :1, :, :, :]

            masks = sliding_window_inference(
                img,
                roi_size=args.rand_crop_size,
                sw_batch_size=2,
                predictor=seg_net,
                mode=args.infer_mode,
                overlap=args.overlap,
            )

            masks = F.interpolate(masks, size=seg.shape[1:], mode="trilinear")
            masks = torch.argmax(masks, dim=1)

            if args.save_pred:
                np.save(
                    os.path.join(save_pred_path, f"{idx}.npy"),
                    masks[0].cpu().numpy().astype(np.int8),
                )

            ssd = surface_distance.compute_surface_distances(
                (seg[0].numpy() == 1),
                (masks[0].cpu().numpy() == 1),
                spacing_mm=spacing[0].numpy(),
            )
            nsd = metrics.compute_surface_dice_at_tolerance(ssd, args.tolerance)  # kits

            masks = F.one_hot(masks, num_classes=args.num_classes).permute(0, 4, 1, 2, 3).float()

            loss = 1 - loss_val(masks, seg.unsqueeze(1).cuda())
            loss_summary.append(loss.detach().cpu().numpy())
            nsd_list.append(nsd)
            logger.info(
                " Case {} - Dice {:.6f} | NSD {:.6f}".format(
                    test_data.dataset.img_dict[idx], loss.item(), nsd
                )
            )
    logger.info(
        "- Dice: {:.6f} - {:.6f} | NSD: {:.6f} - {:.6f}".format(
            np.mean(loss_summary), np.std(loss_summary), np.mean(nsd_list), np.std(nsd_list)
        )
    )


if __name__ == "__main__":
    main()
