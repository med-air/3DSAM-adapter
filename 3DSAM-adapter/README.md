# Training

Type the command below to train the 3DSAM-adapter:
```sh
python train.py --data kits --snapshot_path "path/to/snapshot/" --data_prefix "path/to/data folder/"  --rand_crop_size 128
```
The pre-trained weight of SAM-B can be downloaded [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) 
and shall be put under the folder ckpt. Users with powerful GPUs can also adapt the model with [SAM-L](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) or [SAM-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

To train the automatic version of our 3DSAM-adapter (which removes the prompt encoder and corresponds to the experiment of table 2 in the paper), type the following command in a similar way:
```sh
python train_auto.py --data kits --snapshot_path "path/to/snapshot/" --data_prefix "path/to/data folder/"  --rand_crop_size 256
```
# Evaluation

Type the command below to evaluate the 3DSAM-adapter:
```sh
python test.py --data kits --snapshot_path "path/to/snapshot/" --data_prefix "path/to/data folder/"  --rand_crop_size 128 --num_prompts 1
```
and similarly:
```sh
python test_auto.py --data kits --snapshot_path "path/to/snapshot/" --data_prefix "path/to/data folder/"  --rand_crop_size 256
```
# Pre-trained Checkpoint

Our pretrained checkpoint can be downloaded through [OneDrive](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155187960_link_cuhk_edu_hk/EgSZwTonMG1Cl_PA7wTP5zgBe-DU4K5rb0woDt3i8U22SA?e=0jmfkq).
For all datasets, the crop size is 128.
