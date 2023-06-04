# 3DSAM-adapter: Holistic Adaptation of SAM from 2D to 3D for Promptable Medical Image Segmentation

Implementation for the paper 3DSAM-adapter: Holistic Adaptation of SAM from 2D to 3D for Promptable Medical Image Segmentation.
by Shizhan Gong, [Yuan Zhong](https://yzrealm.com/), Wenao Ma, Jinpeng Li, Zhao Wang, Jingyang Zhang, [Pheng-Ann Heng](https://www.cse.cuhk.edu.hk/~pheng/), and [Qi Dou](https://www.cse.cuhk.edu.hk/~qdou/index.html).

# Sample Results
![Alt text](asset/result.png?raw=true "Title")

# Setup
We recommend using Miniconda to set up an environment:
```
conda create -n med_sam python=3.9.16
conda activate med_sam
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/deepmind/surface-distance.git
pip install -r requirements.txt
```
We managed to test our code on Ubuntu 18.04 with Python 3.9 and CUDA 11.3. Our implementation is based on single GPU setting, but can be easily adapted to use multiple GPUs.

# Dataset
We use the following four open-source datasets for training and evaluation our model

[kITS 2021](https://kits-challenge.org/kits21/)

[MSD-Pancreas](http://medicaldecathlon.com/)

[LiTS 2017](https://competitions.codalab.org/competitions/17094)

[MSD-Colon](http://medicaldecathlon.com/)

The train-validation-test split file can be found in datafile folder. Put the split file under the root directory storing the data.

# Training
Type the command below to train the baselines:
```sh
python train_baselines.py --data kits -m swin_unetr --snapshot_path "path/to/snapshot/" --data_prefix "path/to/data folder/"  --rand_crop_size 128 
```
For training Swin-UNETR, download the [checkpoint](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt) and put it under the folder ckpt.

Type the command below to train the 3DSAM-adapter:
```sh
python train_ours.py --data kits --snapshot_path "path/to/snapshot/" --data_prefix "path/to/data folder/"  --rand_crop_size 256
```
The pre-trained weight of SAM-B can be downloaded [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) 
and shall be put under the folder ckpt. Users with powerful GPUs can also adapt the model with [SAM-L](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) or [SAM-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

# Evaluation
Type the command below to evaluate the performance baselines:
```sh
python test_baseline.py --data kits -m swin_unetr --snapshot_path "path/to/snapshot/" --data_prefix "path/to/data folder/"  --rand_crop_size 128 
```

Type the command below to evaluate the 3DSAM-adapter:
```sh
python test_ours.py --data kits --snapshot_path "path/to/snapshot/" --data_prefix "path/to/data folder/"  --rand_crop_size 256 --num_prompts 1
```

# Pre-trained Checkpoint

Our pretrained checkpoint can be downloaded through [one-drive](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155187960_link_cuhk_edu_hk/EgSZwTonMG1Cl_PA7wTP5zgBe-DU4K5rb0woDt3i8U22SA?e=0jmfkq).
For KiTS, LiTS and MSD-Colon, the crop size is 256. For MSD-Pancreas, the crop size is 128.

## Acknowledgement
Our code is based on [Segment-Anything](https://github.com/facebookresearch/segment-anything).

## Contact
For any questions, please contact <a href="mailto:szgong22@cse.cuhk.edu.hk">szgong22@cse.cuhk.edu.hk</a>
