# CIFAR100   FLOWERS102&CIFAR10
# gfnet-b        gfnet-xs             00
# gfnet-s-d      gfnet-ti-d           01
# dynn                            10 and 11
#                                 jei-dnn
#                                 ViT
python -m kernprof -l -v test.py \
--data-set FLOWERS --arch dynn --method 11 \
--data-path Your_path/Data/CIFAR10 \
--model_path Your_path/model_path \
# --eval


