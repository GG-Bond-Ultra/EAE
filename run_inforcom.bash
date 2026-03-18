# CIFAR100   FLOWERS102
# gfnet-b    gfnet-xs     00
# gfnet-s-d  gfnet-ti-d   01
# dynn                    10 and 11
#                         jei-dnn
#                         ViT
cd /home/nvidianano/heShaoWei/GFNet/
python -m kernprof -l -v new_inforcom.py \
--data-set FLOWERS --arch dynn --method jei-dnn \
--data-path /home/nvidianano/heShaoWei/GFNet/Data/FLOWERS102 \
# --eval


