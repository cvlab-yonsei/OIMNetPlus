# PyTorch Implementation of OIMNet++ (ECCV 2022)
This is an official PyTorch implementation of "OIMNet++: Prototypical Normalization and Localization-aware Learning for Person Search", ECCV 2022.

For more details, visit our [project site](https://cvlab.yonsei.ac.kr/projects/OIMNetPlus/) or see our [paper](http://arxiv.org/abs/2207.10320).<br>
Our main contributions can be found in `models/custom_modules.py` and `losses/oim.py`.<br>

## Requirements
* Python 3.8
* PyTorch 1.7.1
* GPU memory >= 22GB

## Features
* Re-implementation of vanilla [OIMNet](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xiao_Joint_Detection_and_CVPR_2017_paper.pdf) in pure [PyTorch](https://pytorch.org/)
* Using [automatic mixed precision](https://pytorch.org/docs/stable/notes/amp_examples.html) to train with larger batch size under limited GPU memory and accelerate training

## Getting Started
First, clone our git repository.

### Docker
We highly recommend using our [Dockerfile](https://github.com/cvlab-yonsei/OIMNetPlus/blob/main/Dockerfile) to set up the environment.
```
# build docker image
$ docker build -t oimnetplus:latest . 

# execute docker container
$ docker run --ipc=host -it -v <working_dir>:/workspace/work -v <dataset_dir>:/workspace/dataset -w /workspace/work oimnetplus:latest /bin/bash 
```

### Prepare datasets
Download [PRW](https://github.com/liangzheng06/PRW-baseline) and [CUHK-SYSU](https://github.com/ShuangLI59/person_search) datasets.<br>
Modify the dataset directories below if necessary.

* PRW: L4 of [configs/prw.yaml](https://github.com/cvlab-yonsei/OIMNetPlus/blob/main/configs/prw.yaml)<br>
* CUHK-SYSU: L3 of [configs/ssm.yaml](https://github.com/cvlab-yonsei/OIMNetPlus/blob/main/configs/ssm.yaml)<br>

Your directories should look like:
```
    <working_dir>
    OIMNetPlus
    ├── configs/
    ├── datasets/
    ├── engines/
    ├── losses/
    ├── models/
    ├── utils/
    ├── defaults.py
    ├── Dockerfile
    └── train.py
    
    <dataset_dir>
    ├── CUHK-SYSU/
    │   ├── annotation/
    │   ├── Image/
    │   └── ...
    └── PRW-v16.04.20/
        ├── annotations/
        ├── frames/
        ├── query_box/
        └── ...
```

## Training and Evaluation

By running the commands below, evaluation results and training losses will be logged into a .txt file in the output directory.

* OIMNet++<br> 
    `$ python train.py --cfg configs/prw.yaml`<br>
    `$ python train.py --cfg configs/ssm.yaml` 

* OIMNet+++<br>
    `$ python train.py --cfg configs/prw.yaml MODEL.ROI_HEAD.AUGMENT True`<br>
    `$ python train.py --cfg configs/ssm.yaml MODEL.ROI_HEAD.AUGMENT True`

* OIMNet<br>
    `$ python train.py --cfg configs/prw.yaml MODEL.ROI_HEAD.NORM_TYPE 'none' MODEL.LOSS.TYPE 'OIM'`<br> 
    `$ python train.py --cfg configs/ssm.yaml MODEL.ROI_HEAD.NORM_TYPE 'none' MODEL.LOSS.TYPE 'OIM'` 

> We support training/evaluation using **single** GPU only. <br>
> This is due to unsynchronized items across multiple GPUs in OIM loss (i.e., LUT and CQ) and ProtoNorm. <br>
> (PRs are always welcomed!)

## Pretrained Models

We provide pretrained weights and the correponding configs below.<br>

|           |   OIMNet++   |   OIMNet+++  |
|-----------|:------------:|:------------:|
| PRW       | [model](https://github.com/cvlab-yonsei/OIMNetPlus/releases/download/v0.1/prw-loimeps0.1-normtypeprotonorm-augmentfalse.pth) <br> [config](https://github.com/cvlab-yonsei/OIMNetPlus/releases/download/v0.1/prw-loimeps0.1-normtypeprotonorm-augmentfalse.yaml) | [model](https://github.com/cvlab-yonsei/OIMNetPlus/releases/download/v0.1/prw-loimeps0.1-normtypeprotonorm-augmenttrue.pth) <br> [config](https://github.com/cvlab-yonsei/OIMNetPlus/releases/download/v0.1/prw-loimeps0.1-normtypeprotonorm-augmenttrue.yaml) |
| CUHK-SYSU | [model](https://github.com/cvlab-yonsei/OIMNetPlus/releases/download/v0.1/ssm-loimeps0.1-normtypeprotonorm-augmentfalse.pth) <br> [config](https://github.com/cvlab-yonsei/OIMNetPlus/releases/download/v0.1/ssm-loimeps0.1-normtypeprotonorm-augmentfalse.yaml) | [model](https://github.com/cvlab-yonsei/OIMNetPlus/releases/download/v0.1/ssm-loimeps0.1-normtypeprotonorm-augmenttrue.pth) <br> [config](https://github.com/cvlab-yonsei/OIMNetPlus/releases/download/v0.1/ssm-loimeps0.1-normtypeprotonorm-augmenttrue.yaml) |


## Citation
```
@inproceedings{lee2022oimnet++,
  title={OIMNet++: Prototypical Normalization and Localization-Aware Learning for Person Search},
  author={Lee, Sanghoon and Oh, Youngmin and Baek, Donghyeon and Lee, Junghyup and Ham, Bumsub},
  booktitle={European Conference on Computer Vision},
  pages={621--637},
  year={2022},
  organization={Springer}
}
```


## Credits
Our person search implementation is heavily based on [Di Chen](https://di-chen.me/)'s [NAE](https://github.com/dichen-cd/NAE4PS) and [Zhengjia Li](https://github.com/serend1p1ty)'s [SeqNet](https://github.com/serend1p1ty/SeqNet).<br>
ProtoNorm implementation is based on [ptrblck](https://github.com/ptrblck)'s manual BatchNorm implementation [here](https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py).
