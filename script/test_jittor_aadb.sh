# Script for Testing
ARCH='resnet50_HLAGCN'
VERSION="v0"
DATASET="aadb"
python -m utils_jittor.eval -b 64 \
       --dataset ${DATASET} --dataroot "/home/xxx/data/AADB_dataset/" \
       --eval_model "results/${ARCH}_${VERSION}_${DATASET}/checkpoint_epoch_020.pth.tar" 2>&1 | tee -a "logs/${ARCH}_${VERSION}_${DATASET}_result.txt"