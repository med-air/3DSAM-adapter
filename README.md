# 3DSAM-adapter: Holistic Adaptation of SAM from 2D to 3D for Promptable Medical Image Segmentation

Implementation for the paper 3DSAM-adapter: Holistic Adaptation of SAM from 2D to 3D for Promptable Medical Image Segmentation.
by Shizhan Gong, [Yuan Zhong](https://yzrealm.com/), Wenao Ma, Jinpeng Li, Zhao Wang, Jingyang Zhang, [Pheng-Ann Heng](https://www.cse.cuhk.edu.hk/~pheng/), and [Qi Dou](https://www.cse.cuhk.edu.hk/~qdou/index.html).

## Sample Results
![Alt text](asset/result.png?raw=true "Title")

## Setup
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

## Dataset
We use the following four open-source datasets for training and evaluation our model: [KiTS 2021](https://kits-challenge.org/kits21/),
[MSD-Pancreas](http://medicaldecathlon.com/),
[LiTS 2017](https://competitions.codalab.org/competitions/17094),
[MSD-Colon](http://medicaldecathlon.com/).

The train-validation-test split file can be found in datafile folder. Put the split file under the root directory storing the data.

Readers can also download the dataset from our [OneDrive](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155187960_link_cuhk_edu_hk/EmsBt_N3vh9CnzO9Wgg0_Q0BqBi9nkZY8PTBRCm0FH22ng?e=znsQVt).

## Usage
Please refer to [3DSAM-adapter](/3DSAM-adapter/) and [baselines](/baselines/) for training and evaluation details.

## Acknowledgement
Our code is based on [Segment-Anything](https://github.com/facebookresearch/segment-anything).

## Contact
For any questions, please contact <a href="mailto:szgong22@cse.cuhk.edu.hk">szgong22@cse.cuhk.edu.hk</a>
