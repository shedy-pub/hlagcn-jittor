# Script for multiple GPU Training
# Please make sure the number of MPI processes is equal the GPU number.
# For example, argument -np should be 4 for 4 GPUs.
ARCH='resnet50_HLAGCN'
VERSION="v0"
DATASET="ava"
CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun -np 4 python -m utils_jittor.train_jittor --arch ${ARCH} \
        --epochs 20 -b 100 --lr 1e-2 --period 2 \
        --dataset ${DATASET} --dataroot "/home/xxx/data/AVA_dataset/" \
        --weight_dir "results/${ARCH}_${VERSION}_${DATASET}" 2>&1 | tee -a "logs/${ARCH}_${VERSION}_${DATASET}.txt"
        
        
# Script for single GPU Training
# ARCH='resnet50_HLAGCN'
# VERSION="v0"
# DATASET="ava"
# CUDA_VISIBLE_DEVICES=0 python -m utils_jittor.train_jittor --arch ${ARCH} \
#         --epochs 20 -b 100 --lr 1e-2 --period 4 \
#         --dataset ${DATASET} --dataroot "/home/xxx/data/AVA_dataset/" \
#         --weight_dir "results/${ARCH}_${VERSION}_${DATASET}" 2>&1 | tee -a "logs/${ARCH}_${VERSION}_${DATASET}.txt"