import os
from dataset.datasets import load_data_volume
import argparse
from torch.optim import AdamW
import numpy as np
import logging
from utils.script_util import save_checkpoint, set_default_arguments
from utils.model_util import get_model
from monai.losses import DiceCELoss, DiceLoss
from monai.inferers import sliding_window_inference
import torch
from utils.util import setup_logger


def set_default_arguments(args):
    args.rand_crop_size = (64, 160, 160)

    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default=None, type=str, choices=["kits", "pancreas", "lits", "colon"]
    )
    parser.add_argument(
        "--snapshot_path",
        default="",
        type=str,
    )
    parser.add_argument(
        "--data_prefix",
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
    parser.add_argument("-bs", "--batch_size", default=2, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument("--max_epoch", default=400, type=int)
    parser.add_argument("--eval_interval", default=4, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_worker", default=6, type=int)

    args = parser.parse_args()
    args = set_default_arguments(args)
    args.snapshot_path = os.path.join(args.snapshot_path, args.data, args.method)
    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)

    setup_logger(logger_name="train", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"train")
    logger.info(str(args))

    train_data = load_data_volume(
        data=args.data,
        batch_size=args.batch_size,
        path_prefix=args.data_prefix,
        augmentation=True,
        split="train",
        rand_crop_spatial_size=args.rand_crop_size,
        convert_to_sam=False,
        num_worker=args.num_worker,
    )
    val_data = load_data_volume(
        data=args.data,
        batch_size=1,
        path_prefix=args.data_prefix,
        augmentation=False,
        split="val",
        rand_crop_spatial_size=args.rand_crop_size,
        do_val_crop=False,
        convert_to_sam=False,
        num_worker=args.num_worker,
        deterministic=True,
    )

    seg_net = get_model(args).cuda()

    seg_net_opt = AdamW(seg_net.parameters(), lr=args.lr, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        seg_net_opt, start_factor=1.0, end_factor=0.01, total_iters=400
    )

    loss_cal = DiceCELoss(
        softmax=True, lambda_dice=1, lambda_ce=1, to_onehot_y=True, include_background=False
    )
    loss_val = DiceLoss(softmax=True, include_background=False, reduction="none", to_onehot_y=True)

    start_epoch = 0
    best_loss = np.inf
    if args.resume:
        ckpt = torch.load(os.path.join(args.snapshot_path, "best.pth.tar"))

        start_epoch = ckpt["epoch"]
        best_loss = ckpt["best_val_loss"]
        seg_net.load_state_dict(ckpt["network_dict"])
        seg_net_opt.load_state_dict(ckpt["opt_dict"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler_dict"])
        logger.info(f"Resume training from epoch {start_epoch}!")
        del ckpt
        torch.cuda.empty_cache()
    else:
        if args.method == "swin_unetr":
            weight = torch.load("ckpt/model_swinvit.pt")
            seg_net.load_from(weights=weight)
            logger.info("Using pretrained self-supervied Swin UNETR backbone weights !")

    for epoch_num in range(start_epoch, args.max_epoch):
        loss_summary = []
        for idx, (img, seg, _) in enumerate(train_data):
            seg_net.train()
            img = img[:, :1, :, :, :]
            img = img.cuda().float()
            masks = seg_net(img)
            loss = loss_cal(masks, seg.unsqueeze(1).cuda())
            loss_summary.append(loss.detach().cpu().numpy())
            seg_net_opt.zero_grad()
            loss.backward()
            seg_net_opt.step()

        logger.info(
            "- [epoch {}] lr: {:.6f} Train metrics: {:.6f}.".format(
                epoch_num, lr_scheduler.get_last_lr()[0], np.mean(loss_summary)
            )
        )
        lr_scheduler.step()

        if (epoch_num + 1) % args.eval_interval == 0:
            with torch.no_grad():
                loss_summary = []
                seg_net.eval()
                for idx, (img, seg, _) in enumerate(val_data):
                    img = img.cuda().float()
                    img = img[:, :1, :, :, :]
                    masks = sliding_window_inference(
                        img,
                        roi_size=args.rand_crop_size,
                        sw_batch_size=2,
                        predictor=seg_net,
                        mode="gaussian",
                    )
                    print(masks.argmax(1).sum())
                    print(seg.sum())
                    loss = loss_val(masks, seg.unsqueeze(1).cuda())
                    loss_summary.append(loss.detach().cpu().numpy())
            logger.info(f"- [epoch {epoch_num}] Val metrics: " + str(np.mean(loss_summary)))

            is_best = False
            if np.mean(loss_summary) < best_loss:
                best_loss = np.mean(loss_summary)
                is_best = True
            save_checkpoint(
                {
                    "epoch": epoch_num + 1,
                    "best_val_loss": best_loss,
                    "network_dict": seg_net.state_dict(),
                    "opt_dict": seg_net_opt.state_dict(),
                    "lr_scheduler_dict": lr_scheduler.state_dict(),
                },
                is_best=is_best,
                checkpoint=args.snapshot_path,
            )
            logger.info("- Val metrics best: " + str(best_loss))


if __name__ == "__main__":
    main()
