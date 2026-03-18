
import argparse
import time
import torch

from dynn.op_counter import measure_model_and_assign_cost_per_exit
from data_loading.data_loader_helper import get_path_to_project_root
from dynn.classifier_training_helper import LossContributionMode
from dynn.gate.gate import GateType  
from dynn.gate_training_helper import GateObjective
from gfnet import GFNet
import numpy as np
from datasets import build_dataset
import utils
from dynn.gfnet_dynn import GFNet_xs_dynn
from functools import partial
import torch.nn as nn
from engine import evaluate
from vit_pytorch import ViT
import tqdm
import torch.profiler

parser = argparse.ArgumentParser(
    description='PyTorch CIFAR10/CIFAR100 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--arch', type=str,
                     # baseline is to train only with warmup, no gating
                    default='gfnet-h-b', help='model to train'
                    )
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--min-lr',default=2e-4,type=float,help='minimal learning rate')
parser.add_argument('--input-size', default=224, type=int, help='images input size')
# Dataset parameters
parser.add_argument('--data-path', default='Your path/datasets', type=str,
                        help='dataset path')
parser.add_argument('--data-set', default='CIFAR10', choices=['CIFAR10','CIFAR100', 'FLOWERS'],
                        type=str, help='Image Net dataset path')
parser.add_argument('--method', default='00', choices=['00', '01', '10', '11', 'jei-dnn', 'ViT'],
                        type=str, help='method, GFNet:00, GFNet+distil:01, GFNet+dynn:10, GFNet+distil+dynn:11')
parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

# Augmentation parameters
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
parser.add_argument('--repeated-aug', action='store_true')
parser.set_defaults(repeated_aug=False)
parser.add_argument('--dist-eval', action='store_true', default=True, help='Enabling distributed evaluation')
# * Random Erase params
parser.add_argument('--reprob', type=float, default=0, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
parser.add_argument('--num_workers', default=10, type=int)
parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--batch', type=int, default=64, help='batch size')
parser.add_argument('--ce_ic_tradeoff',default=0.7,type=float,help='cost inference and cross entropy loss tradeoff')
parser.add_argument('--num_epoch', default=100, type=int, help='num of epochs')
parser.add_argument('--max_warmup_epoch', default=5, type=int, help='max num of warmup epochs')
parser.add_argument('--bilevel_batch_count',default=200,type=int,help='number of batches before switching the training modes')
parser.add_argument('--barely_train',action='store_true',help='not a real run')
parser.add_argument('--resume', '-r',action='store_true',help='resume from checkpoint')
parser.add_argument('--gate',type=GateType,default=GateType.UNCERTAINTY,choices=GateType)
parser.add_argument('--drop-path',type=float,default=0.1,metavar='PCT',help='Drop path rate (default: None)')
parser.add_argument('--gate_objective', type=GateObjective, default=GateObjective.CrossEntropy, choices=GateObjective)
parser.add_argument('--transfer-ratio',type=float,default=0.01, help='lr ratio between classifier and backbone in transfer learning')
parser.add_argument('--proj_dim',default=32,help='Target dimension of random projection for ReLU codes')
parser.add_argument('--num_proj',default=16,help='Target number of random projection for ReLU codes')
parser.add_argument('--use_mlflow',default=True, help='Store the run with mlflow')
parser.add_argument('--classifier_loss', type=LossContributionMode, default=LossContributionMode.BOOSTED, choices=LossContributionMode)
parser.add_argument('--early_exit_warmup', default=True)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--model_path', default='Your path/model_path', type=str,
                        help='model weight path')
args = parser.parse_args()

seed = args.seed + utils.get_rank()
print("seed:",seed)
torch.manual_seed(seed)
np.random.seed(seed)

if args.barely_train:
    print(
        '++++++++++++++WARNING++++++++++++++ you are barely training to test some things'
    )

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")
# device = 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
path_project = get_path_to_project_root()
model = args.arch


IMG_SIZE = args.input_size
dataset_train, NUM_CLASSES = build_dataset(is_train=True, args=args)
dataset_val, _ = build_dataset(is_train=False, args=args)

sampler_train = torch.utils.data.RandomSampler(dataset_train)
sampler_val = torch.utils.data.SequentialSampler(dataset_val)
data_loader_train = torch.utils.data.DataLoader(
    dataset_train, sampler=sampler_train,
    batch_size=args.batch,
    num_workers=args.num_workers,
    pin_memory=args.pin_mem,
    drop_last=True,
)

data_loader_val = torch.utils.data.DataLoader(
    dataset_val, sampler=sampler_val,
    batch_size=1,
    num_workers=0,
    pin_memory=args.pin_mem,
    drop_last=False
)

path = args.model_path

print('==> Building model..')
is_dynn ,is_distillation = False, False
if args.method == "00":
    is_dynn ,is_distillation = False, False
elif args.method == "01":
    is_dynn ,is_distillation = False, True
elif args.method == "10":
    is_dynn ,is_distillation = True, False
elif args.method in ["11", "jei-dnn"]:
    is_dynn ,is_distillation = True, True

if args.data_set == 'CIFAR100':
    threshold = 0.55
    if is_dynn == True and is_distillation == False: # ablation-dynn
        threshold = 0.5
    if args.method == "jei-dnn":
        threshold = 0.8
    
       
elif args.data_set == 'FLOWERS':
    threshold = 0.25
    if is_dynn == True and is_distillation == False: # ablation-dynn
        threshold = 0.3
    if args.method == "jei-dnn":
        threshold = 0.8

        
elif args.data_set == 'CIFAR10':
    threshold = 0.75
    if is_dynn == True and is_distillation == False: # ablation-dynn
        threshold = 0.7
    if args.method == "jei-dnn":
        threshold = 0.9

if is_dynn == True and is_distillation == True:  
    if args.data_set in ['CIFAR10', 'FLOWERS']:
        ## gfnet-ti-D-dynn
        model = GFNet_xs_dynn(
                img_size=IMG_SIZE, num_classes=NUM_CLASSES, threshold=threshold,
                patch_size=16, embed_dim=256, depth=12, mlp_ratio=4,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), distil=False
        )
        transformer_layer_gating = [5, 7, 9]
        layers = 12
    
    elif args.data_set == 'CIFAR100' :
        ## gfnet-s-d-dynn
        model = GFNet_xs_dynn(
                img_size=IMG_SIZE, num_classes=NUM_CLASSES, threshold=threshold,
                patch_size=16, embed_dim=384, depth=19, mlp_ratio=4,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), distil=False, 
                teacher_embed_dim=512
            )
        transformer_layer_gating = [10, 12, 14, 16]
        layers = 19
    ## jei-dnn
    if args.method == "jei-dnn":
        transformer_layer_gating = [g for g in range(layers - 1)]    
        args.G = len(transformer_layer_gating)


elif is_dynn == True and is_distillation == False: 
    if args.data_set in ['CIFAR10', 'FLOWERS']:
        ## gfnet-xs-dynn
        model = GFNet_xs_dynn(
                img_size=IMG_SIZE, num_classes=NUM_CLASSES, threshold=threshold,
                patch_size=16, embed_dim=384, depth=12, mlp_ratio=4,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), distil=False
        )
        transformer_layer_gating = [5, 7, 9]
        layers = 12
    elif args.data_set == 'CIFAR100' :
        ## gfnet-b-dynn
        model = GFNet_xs_dynn(
            img_size=IMG_SIZE, num_classes=NUM_CLASSES, threshold=threshold,
            patch_size=16, embed_dim=512, depth=19, mlp_ratio=4, drop_path_rate=0.25,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), distil=False
        )
        transformer_layer_gating = [8, 10, 12, 14, 16]
        layers = 19

else: 
    if args.arch == 'gfnet-ti-d':
        model = GFNet(
            img_size=IMG_SIZE, num_classes=NUM_CLASSES,
            patch_size=16, embed_dim=256, depth=12, mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), distil=True, 
            teacher_embed_dim=384
        )
    elif args.arch == 'gfnet-xs':    
        model = GFNet(
            img_size=IMG_SIZE, num_classes=NUM_CLASSES,
            patch_size=16, embed_dim=384, depth=12, mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
    elif args.arch == 'gfnet-b':
        model = GFNet(
            img_size=IMG_SIZE, num_classes=NUM_CLASSES,
            patch_size=16, embed_dim=512, depth=19, mlp_ratio=4, drop_path_rate=0.25,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    elif args.arch == 'gfnet-s-d':
        model = GFNet(
            img_size=IMG_SIZE, num_classes=NUM_CLASSES,
            patch_size=16, embed_dim=384, depth=19, mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), distil=True, 
            teacher_embed_dim=512
        )


if args.method == 'ViT':
    model = ViT(
        image_size = IMG_SIZE,
        patch_size = 16,
        num_classes = NUM_CLASSES,
        dim = 384,
        depth = 12,
        heads = 16,
        mlp_dim = 1538,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    if args.data_set == 'CIFAR100':
        model = ViT(
            image_size = IMG_SIZE,
            patch_size = 16,
            num_classes = NUM_CLASSES,
            dim = 512,
            depth = 12,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

checkpoint = torch.load(path, map_location=torch.device(device))

if is_dynn == True:       
    model.set_CE_IC_tradeoff(args.ce_ic_tradeoff)
    model.set_intermediate_heads(transformer_layer_gating)

    model.set_learnable_gates(transformer_layer_gating,
                        direct_exit_prob_param=True)

    n_flops, n_params, n_flops_at_gates = measure_model_and_assign_cost_per_exit(model, IMG_SIZE, IMG_SIZE, num_classes=NUM_CLASSES)
    mult_add_at_exits = (torch.tensor(n_flops_at_gates) / 1e6).tolist()
    print(mult_add_at_exits)

model = model.to(device)
model_without_ddp = model

print('==> Resuming from checkpoint..')
print(threshold)
param_with_issues =  model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

name_list = ["gates", "intermediate_heads_dist", "intermediate_heads"]
# name_list = []

for name, module in model.named_modules():
    if any(substring in name for substring in name_list):
        module.to(device_cpu)
    else:
        module.to(device_gpu)
for name, param in model_without_ddp.named_parameters():
    print(f"Parameter name: {name}, device: {param.device}")

total_params = sum(p.numel() for p in model_without_ddp.parameters())
print(f"Total number of parameters: {total_params}")
print("Missing keys:", param_with_issues.missing_keys)
print("Unexpected keys:", param_with_issues.unexpected_keys)

if is_dynn == True and args.eval:
    nums_gate = len(transformer_layer_gating)
    evaluate(data_loader_val, model, device, nums_gate)

best_acc = 0
model.eval()

dummy_input = torch.randn(1, 3,224,224,dtype=torch.float).to(device)
    

print('warm up ...\n')
with torch.no_grad():
    for _ in range(100):
        if args.method == 'ViT':
            _ = model.forward(dummy_input)
        else:
            _ = model.forward_features(dummy_input)

total = len(data_loader_val)
num = 1
for i in range(num):

    print('testing ...\n')
    total_forward_time = 0.0  

    start_event = time.time()
    with torch.no_grad():
        with tqdm.tqdm(total=total) as pbar:
            for batch_idx, (inputs, _) in enumerate(data_loader_val):
                start_forward_time = time.time()
                inputs = inputs.to(device, non_blocking=True)
                if is_dynn == True:
                    _ = model.forward_dynn(inputs)
                # elif is_distillation == True:
                #     _ = model.forward_DISTIL(inputs)
                else:
                    _ = model.forward(inputs)
                end_forward_time = time.time()
                forward_time = end_forward_time - start_forward_time
                total_forward_time += forward_time * 1000  
                pbar.update(1)


    end_event = time.time()
    elapsed_time = (end_event - start_event) 
    fps = total / elapsed_time

    elapsed_time_ms = elapsed_time / total

    avg_forward_time = total_forward_time / total

    print(f"Dataset:{args.data_set}-{i}")
    print(f"Model:{args.arch}")
    print(f"FPS: {fps}")
    print("elapsed_time_ms:", elapsed_time_ms * 1000)
    print(f"Avg Forward Time per Image: {avg_forward_time} ms")

    with open(f'Your path/OUT/output_{args.data_set}.txt', 'a') as file:
        file.write(f"Dataset: {args.data_set}-{i}\n")
        file.write(f"Model: {args.arch}\n")
        file.write(f"Method: {args.method}\n")
        file.write(f"FPS: {fps}\n")
        file.write(f"elapsed_time_ms: {elapsed_time_ms * 1000} ms\n")
        file.write(f"Avg Forward Time per Image: {avg_forward_time} ms\n")
        file.write("=================================================================\n")