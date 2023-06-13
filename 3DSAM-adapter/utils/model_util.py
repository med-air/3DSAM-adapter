def get_model(args):
    if args.method == "swin_unetr":
        from monai.networks.nets import SwinUNETR

        seg_net = SwinUNETR(
            img_size=args.rand_crop_size,
            in_channels=1,
            out_channels=args.num_classes,
            feature_size=48,
            use_checkpoint=True,
        )
    elif args.method == "3d_uxnet":
        from modeling.uxnet import UXNET

        seg_net = UXNET(
            in_chans=1,
            out_chans=args.num_classes,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            spatial_dims=3,
        )

    elif args.method == "unetr++":
        from modeling.unetr_pp.unetr_pp import UNETR_PP

        # if args.data == "msd":
        #     patch_size = [2, 4, 4]
        # else:
        #     patch_size = [4, 4, 4]

        seg_net = UNETR_PP(
            in_channels=1,
            out_channels=args.num_classes,
            img_size=args.rand_crop_size,
            patch_size=[2, 4, 4],
            feature_size=16,
            num_heads=4,
            depths=[3, 3, 3, 3],
            dims=[32, 64, 128, 256],
            do_ds=False,
        )

    elif args.method == "unetr":
        from modeling.unetr import UNETR

        seg_net = UNETR(
            in_channels=1,
            out_channels=args.num_classes,
            img_size=args.rand_crop_size,
            patch_size=16,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        )

    elif args.method == "nnformer":
        from modeling.nnFormer.nnFormer_seg import nnFormer

        seg_net = nnFormer(
            input_channels=1,
            num_classes=args.num_classes,
            crop_size=args.rand_crop_size,
            patch_size=[2, 4, 4],
            window_size=[8, 8, 6, 4],
        )

    elif args.method == "transbts":
        from modeling.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS

        _, seg_net = TransBTS(img_dim=args.rand_crop_size, num_classes=args.num_classes)

    else:
        raise NotImplementedError

    return seg_net
