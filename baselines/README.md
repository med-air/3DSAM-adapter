# Training
Type the command below to train the baselines:
```sh
python train.py --data kits -m swin_unetr --snapshot_path "path/to/snapshot/" --data_prefix "path/to/data folder/"
```
Using `-m` to indicate the method. We compared with [nnUNet](https://arxiv.org/abs/1809.10486), [TransBTS](https://arxiv.org/abs/2103.04430), [nnFormer](https://arxiv.org/abs/2109.03201), [Swin UNETR](https://arxiv.org/abs/2111.14791), [UNETR++](https://arxiv.org/abs/2212.04497), and [3D UX-Net](https://arxiv.org/abs/2209.15076). 

For training Swin-UNETR, download the [checkpoint](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt) and put it under the folder ckpt.

We use various hyper-parameters for each dataset, for more details, please refer to [datasets.py](dataset/datasets.py). The crop size is set as `(64, 160, 160)` for all datasets.

# Evaluation
Type the command below to evaluate the performance baselines:
```sh
python test.py --data kits -m swin_unetr --snapshot_path "path/to/snapshot/" --data_prefix "path/to/data folder/"
```
