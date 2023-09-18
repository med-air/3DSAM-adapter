from dataset.datasets import load_data_volume
import argparse
import numpy as np
import logging
from monai.losses import DiceCELoss, DiceLoss
from modeling.Med_SAM.image_encoder import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
import torch.nn.functional as F
from modeling.Med_SAM.mask_decoder import VIT_MLAHead_h as VIT_MLAHead
import torch
from modeling.Med_SAM.prompt_encoder import PromptEncoder, TwoWayTransformer
from functools import partial
import os
from utils.util import setup_logger
import surface_distance
from surface_distance import metrics
from monai.inferers import sliding_window_inference

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
        "--rand_crop_size",
        default=0,
        nargs='+', type=int,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
    )

    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--num_worker", default=6, type=int)
    parser.add_argument(
        "--checkpoint",
        default="last",
        type=str,
    )
    parser.add_argument("-tolerance", default=5, type=int)
    args = parser.parse_args()
    if args.checkpoint == "last":
        file = "last.pth.tar"
    else:
        file = "best.pth.tar"
    device = args.device
    if args.rand_crop_size == 0:
        if args.data in ["colon", "pancreas", "lits", "kits"]:
            args.rand_crop_size = (256, 256, 256)
    else:
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)
    args.snapshot_path = os.path.join(args.snapshot_path, args.data)

    setup_logger(logger_name="test", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"test")
    logger.info(str(args))
    test_data = load_data_volume(
        data=args.data,
        batch_size=1,
        path_prefix=args.data_prefix,
        augmentation=False,
        split="test",
        rand_crop_spatial_size=args.rand_crop_size,
        convert_to_sam=False,
        do_test_crop=False,
        deterministic=True,
        num_worker=0
    )
    img_encoder = ImageEncoderViT_3d(
        depth=12,
        embed_dim=768,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        cubic_window_size=8,
        out_chans=256,
        num_slice = 16)
    img_encoder.load_state_dict(torch.load(os.path.join(args.snapshot_path, file), map_location='cpu')["encoder_dict"], strict=True)
    img_encoder.to(device)


    mask_decoder = VIT_MLAHead(img_size = 96).to(device)
    mask_decoder.load_state_dict(torch.load(os.path.join(args.snapshot_path, file), map_location='cpu')["decoder_dict"],
                          strict=True)
    mask_decoder.to(device)

    dice_loss = DiceLoss(include_background=False, softmax=False, to_onehot_y=True, reduction="none")
    img_encoder.eval()
    mask_decoder.eval()

    patch_size = args.rand_crop_size[0]

    def model_predict(img, img_encoder, mask_decoder):
        out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode='trilinear')
        input_batch = out[0].transpose(0, 1)
        batch_features, feature_list = img_encoder(input_batch)
        feature_list.append(batch_features)

        new_feature = feature_list
        img_resize = F.interpolate(img[0, 0].permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(device), scale_factor=64/patch_size,
                                   mode="trilinear")
        new_feature.append(img_resize)
        masks = mask_decoder(new_feature, 2, patch_size//64)
        masks = masks.permute(0, 1, 4, 2, 3)
        masks = torch.softmax(masks, dim=1)
        masks = masks[:, 1:]
        return masks

    with torch.no_grad():
        loss_summary = []
        loss_nsd = []
        for idx, (img, seg, spacing) in enumerate(test_data):
            seg = seg.float()
            seg = seg.to(device)
            img = img.to(device)
            pred = sliding_window_inference(img, [256, 256, 256], overlap=0.5, sw_batch_size=1,
                                            mode="gaussian",
                                            predictor=partial(model_predict,
                                                              img_encoder=img_encoder,
                                                              mask_decoder=mask_decoder))
            pred = F.interpolate(pred, size=seg.shape[1:], mode="trilinear")
            seg = seg.unsqueeze(0)
            if torch.max(pred) < 0.5 and torch.max(seg) == 0:
                loss_summary.append(1)
                loss_nsd.append(1)
            else:
                masks = pred > 0.5
                loss = 1 - dice_loss(masks, seg)
                loss_summary.append(loss.detach().cpu().numpy())
                ssd = surface_distance.compute_surface_distances((seg == 1)[0, 0].cpu().numpy(),
                                                                 (masks==1)[0, 0].cpu().numpy(),
                                                                 spacing_mm=spacing[0].numpy())
                nsd = metrics.compute_surface_dice_at_tolerance(ssd, args.tolerance)  # kits
                loss_nsd.append(nsd)
            logger.info(
                " Case {} - Dice {:.6f} | NSD {:.6f}".format(
                    test_data.dataset.img_dict[idx], loss.item(), nsd
                ))
        logging.info("- Test metrics Dice: " + str(np.mean(loss_summary)))
        logging.info("- Test metrics NSD: " + str(np.mean(loss_nsd)))


if __name__ == "__main__":
    main()

