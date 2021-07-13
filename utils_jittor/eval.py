# -*- encoding: utf-8 -*-
# !/usr/bin/env python

import os
import time, datetime
import numpy as np
import jittor
import jittor.transform as transforms
from utils_jittor.dataset import *
from utils_jittor.parameter import *
from utils_jittor.util import *
from utils_jittor.train_jittor import val_test_process, create_network

def main():
    jittor.flags.use_cuda = 1
    args = get_parameters(description='HLAGCN PyTorch Model Testing')
    if not args.arch in model_names:
        raise RuntimeError('Network architecture {} is not supported!'.format(args.arch))
    if not os.path.isfile(args.eval_model):
        raise RuntimeError('Evaluation Model {} does not exist!'.format(args.eval_model))
    else:
        print(" "*5 + "@ [Evaluation Model]")
    main_worker(args)

def main_worker(args):
    if args.dataset == 'ava':
        _, _, _, _, img_paths_test, img_rates_test, img_cls_test, img_ratios_test = preload_img(params_ava, args.dataroot)
        args.bins = 10
        data_test = [img_paths_test, img_rates_test, img_cls_test,img_ratios_test]
    elif args.dataset == 'aadb':
        [_, _, _, _, img_paths_test, img_rates_test, img_cls_test, img_ratios_test, _, _, _, _] = preload_img(params_aadb, args.dataroot)
        data_test = [img_paths_test, img_rates_test, img_cls_test,img_ratios_test]
        args.bins = 5
    else:
        raise RuntimeError('Invalid Dataset => {}'.format(args.dataset))  
    model = create_network(args)
    print(" "*5 + f'@ Test images #{len(img_paths_test)}')
    normalize = transforms.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        normalize])
    test_set = Dataset(data_test, transform=val_transform)   
    params_val_test = {'batch_size': args.batch_size, 
        'shuffle': False,
        'num_workers': args.workers,
        'collate_batch': collate_wrapper,}
    test_loader = test_set.set_attrs(**params_val_test)
    criterions = [_, emd_loss(dist_r=1) ]
    model, _, _ = load_checkpoint(args.eval_model, model, [], strict=True)   
    val_test_process(test_loader, model, criterions, args)


if __name__ == '__main__':
    main()