# Lightweight Dynamic Global Filter Network for Efficient Image Analysis on Edge Devices

Created by Shaowei He, Ziyang Zhang, Shusheng Li, Jie Liu

This repository contains PyTorch implementation for EAE (SECON 2026).

EAE is a holistic Efficient image Analysis framework for Edge devices that synergizes spectral-domain model compression with a hardware-aware dynamic inference mechanism.First, we introduce a novel frequency-domain knowledge distillation technique that regularizes student feature norms and orientations against a teacher network, achieving significant parameter reduction while preserving global receptive fields. Second, we dismantle the "exit-at-every-layer" paradigm common in dynamic networks, which we demonstrate creates prohibitive synchronization overheads for lightweight backbones. Instead, EAE employs a strategic, interval-based exit placement optimized for the specific latency characteristics of embedded GPUs. Furthermore, we propose a heterogeneous orchestration strategy that offloads lightweight gating and uncertainty estimation to the CPU, thereby minimizing GPU kernel interruptions.

<img width="910" height="444" alt="image" src="https://github.com/user-attachments/assets/2b65d9a6-357f-4668-ad36-07df3cd7d47a" />

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

To get the results on ActivityNet1.3, run the following command:

```
python infer.py --data-path /path/to/ILSVRC2012/ --arch arch_name --model-path /path/to/model
```



## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{he2026eae,
  title={Lightweight Dynamic Global Filter Network for Efficient Image Analysis on Edge Devices},
  author={Shaowei He, Ziyang Zhang, Shusheng Li, Jie Liu},
  booktitle = {22nd Annual IEEE International Conference on Sensing, Communication, and Networking 2026 (SECON)},
  year = {2026}
}
```
