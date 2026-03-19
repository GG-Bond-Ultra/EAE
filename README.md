# Lightweight Dynamic Global Filter Network for Efficient Image Analysis on Edge Devices

Created by Shaowei He, Ziyang Zhang, Shusheng Li, Jie Liu

This repository contains PyTorch implementation for EAE (SECON 2026).

EAE is a holistic Efficient image Analysis framework for Edge devices that synergizes spectral-domain model compression with a hardware-aware dynamic inference mechanism.First, we introduce a novel frequency-domain knowledge distillation technique that regularizes student feature norms and orientations against a teacher network, achieving significant parameter reduction while preserving global receptive fields. Second, we dismantle the "exit-at-every-layer" paradigm common in dynamic networks, which we demonstrate creates prohibitive synchronization overheads for lightweight backbones. Instead, EAE employs a strategic, interval-based exit placement optimized for the specific latency characteristics of embedded GPUs. Furthermore, we propose a heterogeneous orchestration strategy that offloads lightweight gating and uncertainty estimation to the CPU, thereby minimizing GPU kernel interruptions.

![intro](EAE.pdf)

Our code is based on [GFNet](https://github.com/raoyongming/GFNet) and [JEI-DNN](https://github.com/networkslab/dynn).


## Usage

### Requirements

- torch>=1.8.0
- torchvision
- timm
- scikit_learn
- fvcore

### Supported datasets：

Out of the box this codebase supports CIFAR10, CIFAR100, FLOWERS102.

### Evaluation

To evaluate a pre-trained GFNet model on the ImageNet validation set with a single GPU, run:

```
python infer.py --data-path /path/to/ILSVRC2012/ --arch arch_name --model-path /path/to/model
```



## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{rao2021global,
  title={Global Filter Networks for Image Classification},
  author={Rao, Yongming and Zhao, Wenliang and Zhu, Zheng and Lu, Jiwen and Zhou, Jie},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2021}
}
```
