# -*- encoding: utf-8 -*-
# !/usr/bin/env python
import argparse
model_names = ['resnet50_HLAGCN'] 
dataset_names = ['ava', 'aadb']

def get_parameters(description='HLA-GCN model params'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    
    # dataset setting  
    parser.add_argument('--dataset', default='ava', choices=dataset_names,
                        help='dataset: ' + ' | '.join(dataset_names) +
                            ' (default: AVA dataset)')
    parser.add_argument('--dataroot', default='',
                        help='dataset path')
    parser.add_argument('--bins', default=10, type=int,
                        metavar='N', help='num of score bins (default: 10)')
    
    # arch setting  
    parser.add_argument('--arch', metavar='ARCH', default='resnet50_HLAGCN', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                            ' (default: resnet50_HLAGCN)')
    parser.add_argument('--imsize', default=300, type=int,
                        metavar='N', help='wrap input image size (default: 300)')
    parser.add_argument('--gcnlayer', default=3, type=int,
                        metavar='N', help='num of the GCN layer in the first LAGCN module (default: 3)')
    parser.add_argument('--config', default=1, type=int,
                        help='arch config')
    
    # training setting 
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate (default: 1e-2)')
    parser.add_argument('-d', '--weight_dir', type=str, default=None, 
                            help='Pass a directory to save weights')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--seed', default=1, type=int,
                        help='random_state')
    parser.add_argument('--shuffle', default=1, type=int,
                        help='shuffle the dataset (default: true)')

    # log setting
    parser.add_argument('--period', default=2, type=int, metavar='N',
                            help="every N epoch save model (default: 2)")
    parser.add_argument('--print_freq', '-p', default=500, type=int,
                        metavar='N', help='print frequency (default: 500)')
    
    # evaluation & validation
    parser.add_argument('--eval_model', type=str, default=None,
                        help='Pass a directory to load weights for evaluation on test set')
    parser.add_argument('--val_freq', default=1, type=int,
                        metavar='N', help='check validation set frequency of epoch (default: 1)')
    parser.add_argument('--resume_model', type=str, default=None, 
                            help='Pass a directory to load weights')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    return parser.parse_args()