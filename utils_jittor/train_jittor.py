# -*- encoding: utf-8 -*-
# !/usr/bin/env python

import os
import time, datetime
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
import jittor
from utils_jittor.dataset import *
from utils_jittor.parameter import *
from utils_jittor.util import *
from jittor.lr_scheduler import MultiStepLR
import jittor.optim
import jittor.transform as transforms
import warnings
warnings.filterwarnings("ignore")

def main():
    jittor.flags.use_cuda = 1
    args = get_parameters(description='HLA-GCN Model Multi-GPU Training')
    if jittor.rank == 0:
        if not args.arch in model_names:
            raise RuntimeError('Network architecture {} is not supported!'.format(args.arch))
        if args.eval_model:
            print(" "*5 + "@ [Evaluation Model]")
        else:
            print(" "*5 +"@ [Training Model] Arch = [{}]; Dataset = [{}]".format(args.arch, args.dataset))
            print(" "*5 +"@ LR = [{}]; Total epoch = [{}]; Batch size = [{}]".format(
                args.lr, args.epochs, args.batch_size))
            print(" "*5 +"@ weight_decay = [{}]; momentum = [{}]; workers = [{}]".format(
                args.weight_decay, args.momentum, args.workers))
            if args.weight_dir is not None:
                print(" "*5 +"@ model save dir: {}".format(args.weight_dir))
                print(" "*5 +"@ model save period: {}".format(args.period))
                if not os.path.exists(args.weight_dir):
                    os.makedirs(args.weight_dir)
            else:
                raise RuntimeError('-d/--weight_dir arguments must be passed as argument')
    main_worker(args)

def main_worker(args):

    normalize = transforms.ImageNormalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((300, 300)),
        transforms.ToTensor(),
        normalize,
        ])
    val_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        normalize])

    # fast debug model
    fast_validation_model = False
    
    # load AVA dataset
    if args.dataset == 'ava':
        img_paths_train_val, img_rates_train_val, img_cls_train_val, img_ratios_train_val,\
        img_paths_test, img_rates_test, img_cls_test, img_ratios_test = preload_img(params_ava, args.dataroot)
        args.bins = 10
        y = img_cls_train_val
        x = np.arange(len(y))
        val_ratio = 2000/len(y)
        train_idx, val_idx, _, _ = train_test_split(x, y, test_size=val_ratio, random_state=args.seed)
        trainval_info = [img_paths_train_val, img_rates_train_val, img_cls_train_val, img_ratios_train_val]
        data_train = [np.array(img_paths_train_val)[train_idx].tolist(),
            np.array(img_rates_train_val)[train_idx].tolist(),
            np.array(img_cls_train_val)[train_idx].tolist(),
            np.array(img_ratios_train_val)[train_idx].tolist(),
            ]
        data_val = [np.array(img_paths_train_val)[val_idx].tolist(),
            np.array(img_rates_train_val)[val_idx].tolist(),
            np.array(img_cls_train_val)[val_idx].tolist(),
            np.array(img_ratios_train_val)[val_idx].tolist(),
            ]
        data_test = [img_paths_test, img_rates_test, img_cls_test,img_ratios_test]
        split_train_num, split_val_num, split_test_num = len(data_train[0]), len(data_val[0]), len(data_test[0])
        if jittor.rank == 0:
            print(" "*5 + f'@ Dataset: train #{split_train_num}, val #{split_val_num}({val_ratio*100:2f}%), test #{split_test_num}')

        if fast_validation_model:
            print('+' * 30)
            print('Fast validation mode: 20% train, 10% val and test')
            print('+' * 30)
            _, train_idx, _, _ = train_test_split(x, y, test_size = 0.2)
            data_train = [list(map(list,zip(*trainval_info)))[i] for i in train_idx]
    
    # load AADB dataset
    elif args.dataset == 'aadb':
        [img_paths_train, img_rates_train, img_cls_train, img_ratios_train, \
               img_paths_test, img_rates_test, img_cls_test, img_ratios_test, \
               img_paths_val, img_rates_val, img_cls_val, img_ratios_val] = preload_img(params_aadb, args.dataroot)
        data_train = [img_paths_test, img_rates_test, img_cls_test,img_ratios_test]
        data_val = [img_paths_val, img_rates_val, img_cls_val,img_ratios_val]
        data_test = [img_paths_test, img_rates_test, img_cls_test,img_ratios_test]
        args.bins = 5
    else:
        raise RuntimeError('Invalid Dataset => {}'.format(args.dataset))

    training_set = Dataset(data_train, transform=train_transform)
    params_train = {'batch_size': args.batch_size,
        'shuffle': True if args.shuffle else False,
        'num_workers': args.workers,
        'collate_batch': collate_wrapper,
        }

    val_set = Dataset(data_val, transform=val_transform)
    test_set = Dataset(data_test, transform=val_transform)
    params_val_test = {'batch_size': args.batch_size//2,
        'shuffle': False,
        'num_workers': args.workers,
        'collate_batch': collate_wrapper,
        }
    train_loader = training_set.set_attrs(**params_train)
    val_loader = val_set.set_attrs(**params_val_test)
    test_loader = test_set.set_attrs(**params_val_test)

    model = create_network(args)
    criterion_aes_train = emd_loss(dist_r=2)
    criterion_aes_val = emd_loss(dist_r=1)
    criterions = [criterion_aes_train, criterion_aes_val]
    
    optimizer = jittor.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    if args.resume_model is not None:
        if not os.path.isfile(args.resume_model):
            raise RuntimeError('Model {} does not exist!'.format(args.resume_model))
        model, _, args.start_epoch, _ = load_checkpoint(args.resume_model, model, [])
    scheduler = MultiStepLR(optimizer, milestones=[8], gamma=0.1)
    
    for epoch in range(args.start_epoch):
        scheduler.step()
    for epoch in range(args.start_epoch+1, args.start_epoch+args.epochs+1):
        if jittor.rank == 0:
            if epoch == args.start_epoch +1:
                print(f'=> Start training #Ep {epoch} /{(args.start_epoch + args.epochs):d}')
            else:
                print(f"#Ep -> {epoch:d}/{args.epochs:d};")
        epoch_start = time.time()
        scheduler.step()
        loss = train(train_loader, model, criterions, optimizer, epoch, args)
        epoch_train_end = time.time()
        tr_time = epoch_train_end - epoch_start
        if jittor.rank == 0:
            print(f'---> Train: {float((tr_time)/60):.2f} min/epoch,',
                f'train loss: {loss:.4f} - lr: {optimizer.lr:.5f}')
            # evaluate on validation set
            if epoch % args.val_freq == 0:
                val_loss, val_acc_aes = val_test_process(val_loader, model, criterions, args)
                
            # saving model
            if epoch % args.period == 0:
                filename = os.path.join(args.weight_dir, 'checkpoint_epoch_{0:03d}.pth.tar'.format(epoch))
                jittor.save({
                    'epoch': epoch,
                    'arch': args.arch,
                    'val': [val_loss, val_acc_aes], 
                    'state_dict': model.state_dict()}, filename)
                print('   * saving model:', filename)
            print('-' * 80)

def train(train_loader, model, criterions, optimizer, epoch, args):
    batch_time = AverageMeter('Net', ':.1f')
    data_time = AverageMeter('Load', ':.1f')
    losses = AverageMeter('loss_avg', ':.3f')
    progress = ProgressMeter(len(train_loader),
        [batch_time, data_time, losses],
        prefix="{}->Ep:[{}]".format(datetime.datetime.now().strftime('%H:%M:%S'), epoch))
    model.train()
    time_end = time.time()
    jittor.sync_all()
    for i, batch_sample in enumerate(train_loader):
        data_time.update(time.time() - time_end)
        input = batch_sample.X1
        target = batch_sample.label
        if 'HLAGCN' in args.arch:
            input_ratio = batch_sample.ratio
            y_pred_aes = model([input,input_ratio])
            y_pred_aes = y_pred_aes[2]
        else:
            y_pred_aes = model(input)
        criterion_aes_train = criterions[0]
        loss = criterion_aes_train(y_pred_aes, target)
        losses.update(loss.item(), input.size(0))
        optimizer.zero_grad()
        optimizer.step(loss)

        batch_time.update(time.time() - time_end)
        time_end = time.time()
        if jittor.rank == 0 and i % args.print_freq == 0:
            progress.display(i)
    return losses.avg

@jittor.single_process_scope()
def val_test_process(eval_loader, model, criterions, args):
    losses_aes, accs_aes = AverageMeter('Loss', ':.3f'), AverageMeter('Acc', ':.2f')
    model.eval()
    with jittor.no_grad():
        scores_hist, labels_hist = [], []
        jittor.sync_all()
        for i, batch_sample in enumerate(eval_loader):
            input = batch_sample.X1
            target = batch_sample.label
            if 'HLAGCN' in args.arch:
                input_ratio = batch_sample.ratio
                y_pred_aes = model([input,input_ratio]) 
                _, _, prediction = y_pred_aes
            else:
                y_pred_aes = model(input)
            criterion_aes_val = criterions[1]
            labels_hist.append(target.numpy())
            scores_hist.append(prediction.numpy())
            loss_aes = criterion_aes_val(prediction, target)
            acc_aes = binary_accuracy(prediction, target, args.bins)
            losses_aes.update(loss_aes.item(), input.size(0))
            accs_aes.update(acc_aes.item(), input.size(0))
    metrics = cal_metrics(scores_hist, labels_hist, args.bins)
    print(f' --> Validation:')
    print(f'     - MSE {metrics[0]:.4f} | SRCC {metrics[1]:.4f} | LCC {metrics[2]:.4f} ')
    print(f'     - Acc {metrics[3]:.2f} | EMD_1 {metrics[4]:.4f}| EMD_2 {metrics[5]:.4f}')
    return [losses_aes.avg, accs_aes.avg]

def create_network(args):
    if args.arch == 'resnet50_HLAGCN':
        from utils_jittor.model import resnet50hlagcn
        model = resnet50hlagcn(num_classes=args.bins, gcn_num=args.gcnlayer)
    return model

if __name__ == '__main__':
    main()
