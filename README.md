# 3DSAM-adapter: Holistic Adaptation of SAM from 2D to 3D for Promptable Medical Image Segmentation

Implementation for the paper [3DSAM-adapter: Holistic Adaptation of SAM from 2D to 3D for Promptable Medical Image Segmentation](https://arxiv.org/pdf/2306.13465.pdf)
by Shizhan Gong, [Yuan Zhong](https://yzrealm.com/), Wenao Ma, Jinpeng Li, Zhao Wang, Jingyang Zhang, [Pheng-Ann Heng](https://www.cse.cuhk.edu.hk/~pheng/), and [Qi Dou](https://www.cse.cuhk.edu.hk/~qdou/index.html).
![Alt text](asset/teaser.png?raw=true "Title")
## Details
> Despite that the segment anything model (SAM) achieved impressive results on
general-purpose semantic segmentation with strong generalization ability on daily
images, its demonstrated performance on medical image segmentation is less
precise and not stable, especially when dealing with tumor segmentation tasks that
involve objects of small sizes, irregular shapes, and low contrast. Notably, the
original SAM architecture is designed for 2D natural images, therefore would not be
able to extract the 3D spatial information from volumetric medical data effectively.
In this paper, we propose a novel adaptation method for transferring SAM from 2D
to 3D for promptable medical image segmentation. Through a holistically designed
scheme for architecture modification, we transfer the SAM to support volumetric
inputs while retaining the majority of its pre-trained parameters for reuse. The
fine-tuning process is conducted in a parameter-efficient manner, wherein most
of the pre-trained parameters remain frozen, and only a few lightweight spatial
adapters are introduced and tuned. Regardless of the domain gap between natural
and medical data and the disparity in the spatial arrangement between 2D and
3D, the transformer trained on natural images can effectively capture the spatial
patterns present in volumetric medical images with only lightweight adaptations.
We conduct experiments on four open-source tumor segmentation datasets, and
with a single click prompt, our model can outperform domain state-of-the-art
medical image segmentation models on 3 out of 4 tasks, specifically by 8.25%,
29.87%, and 10.11% for kidney tumor, pancreas tumor, colon cancer segmentation,
and achieve similar performance for liver tumor segmentation. We also compare
our adaptation method with existing popular adapters, and observed significant
performance improvement on most datasets.

## Datasets
![Alt text](asset/datasets.png?raw=true "Title")
We use the 4 open-source datasets for training and evaluation our model.
-  KiTS 2021 [[original paper]](https://www.sciencedirect.com/science/article/abs/pii/S1361841520301857)[[original dataset]](https://kits-challenge.org/kits21/)[[our preprocessed dataset]](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155187960_link_cuhk_edu_hk/Ebe8F12v_JtOv2ovW3a-BjkB8LryC6BFZZwtsi0kAikphw?e=w728Ud)
- MSD-Pancreas [[original paper]](https://www.nature.com/articles/s41467-022-30695-9)[[original dataset]](http://medicaldecathlon.com/)[[our preprocessed dataset]](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155187960_link_cuhk_edu_hk/EdH84TX9CJ5CiXUjIyeXEZ4B-6AK8LfLhIhlIfiVDicfVQ?e=avTPPf)
- LiTS 2017 [[original paper]](https://www.sciencedirect.com/science/article/pii/S1361841522003085)[[original dataset]](https://competitions.codalab.org/competitions/17094)[[our preprocessed dataset]](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155187960_link_cuhk_edu_hk/EcqXHRupWoxNjYkmoiHQl4QBpvTS41TfJfqfO0x0xOxgow?e=ueD0i2)
- MSD-Colon [[original paper]](https://www.nature.com/articles/s41467-022-30695-9)[[original dataset]](http://medicaldecathlon.com/)[[our preprocessed dataset]](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155187960_link_cuhk_edu_hk/EX0cgfQJykZCiY7QAFwp-BUBR349boTd0noDU8VxkGHiEw?e=kwp893)


## Sample Results
![Alt text](asset/result.png?raw=true "Title")

## Get Started

#### Main Requirements
- python=3.9.16
- cuda=11.3
- torch==1.12.1
- torchvision=0.13.1
#### Installation
We suggest using Anaconda to setup environment on Linux, if you have installed anaconda, you can skip this step.
```sh
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh && zsh Anaconda3-2020.11-Linux-x86_64.sh
```
Then, we can create environment and install packages using provided `requirements.txt`
```sh
conda create -n med_sam python=3.9.16
conda activate med_sam
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/deepmind/surface-distance.git
pip install -r requirements.txt
```
Our implementation is based on single GPU setting (NVIDIA A40 GPU), but can be easily adapted to use multiple GPUs. We need about 35GB of memory to run.

#### 3DSAM-adapter (Ours)
To use the code, first go to the folder `3DSAM-adapter`
```sh
cd 3DSAM-adapter
```
Type the command below to train the 3DSAM-adapter:
```sh
python train.py --data kits --snapshot_path "path/to/snapshot/" --data_prefix "path/to/data folder/" 
```
The pre-trained weight of SAM-B can be downloaded [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) 
and shall be put under the folder `ckpt`. Users with powerful GPUs can also adapt the model with [SAM-L](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) or [SAM-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

Type the command below to evaluate the 3DSAM-adapter:
```sh
python test.py --data kits --snapshot_path "path/to/snapshot/" --data_prefix "path/to/data folder/"  --num_prompts 1
```
Using  `--num_prompts` to indicate the number of points used as prompt, the default value is 1.

Our pretrained checkpoint can be downloaded through [OneDrive](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155187960_link_cuhk_edu_hk/EgSZwTonMG1Cl_PA7wTP5zgBe-DU4K5rb0woDt3i8U22SA?e=0jmfkq).
For all four datasets, the crop size  is 128.

#### Baselines

We provide our implementation for baselines includes

- Swin UNETR [[original paper]](https://arxiv.org/abs/2111.14791)[[original implementation]](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR)
- 3D UX-Net [[original paper]](https://arxiv.org/abs/2209.15076)[[original implementation]](https://github.com/MASILab/3DUX-Net)
- UNETR++ [[original paper]](https://arxiv.org/abs/2212.04497)[[original implementation]](https://github.com/Amshaker/unetr_plus_plus)
- TransBTS [[original paper]](https://arxiv.org/abs/2103.04430)[[original implementation]](https://github.com/Wenxuan-1119/TransBTS)
- nnFormer [[original paper]](https://arxiv.org/abs/2109.03201)[[original implementation]](https://github.com/282857341/nnFormer)

To use the code, first go to the folder `baselines`

```sh
cd baselines
```

Type the command below to train the baselines:

```sh
python train.py --data kits -m swin_unetr --snapshot_path "path/to/snapshot/" --data_prefix "path/to/data folder/"
```

Using  `--data` to indicate the dataset, can be one of `kits`, `pancreas`, `lits`, `colon`

Using `-m` to indicate the method, can be one of `swin_unetr`, `unetr`, `3d_uxnet`, `nnformer`, `unetr++`, `transbts`

For training Swin-UNETR, download the [checkpoint](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt) and put it under the folder ckpt.

We use various hyper-parameters for each dataset, for more details, please refer to [datasets.py](dataset/datasets.py). The crop size is set as `(64, 160, 160)` for all datasets.

Type the command below to evaluate the performance baselines:

```sh
python test.py --data kits -m swin_unetr --snapshot_path "path/to/snapshot/" --data_prefix "path/to/data folder/"
```

## Feedback and Contact
For any questions, please contact <a href="mailto:szgong22@cse.cuhk.edu.hk">szgong22@cse.cuhk.edu.hk</a>

## Acknowledgement
Our code is based on [Segment-Anything](https://github.com/facebookresearch/segment-anything), [3D UX-Net](https://github.com/MASILab/3DUX-Net), and [Swin UNETR](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb).

## Citation
If you find this code useful, please cite in your research papers.
```
@article{Gong20233DSAMadapterHA,
  title={3DSAM-adapter: Holistic Adaptation of SAM from 2D to 3D for Promptable Medical Image Segmentation},
  author={Shizhan Gong and Yuan Zhong and Wenao Ma and Jinpeng Li and Zhao Wang and Jingyang Zhang and Pheng-Ann Heng and Qi Dou},
  journal={arXiv preprint arXiv:2306.13465},
  year={2023}
}
```
