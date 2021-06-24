# -*- encoding: utf-8 -*-
import os
import jittor
import jittor.nn
import jittor.optim
import numpy as np
import time
from scipy.stats import pearsonr, spearmanr

class emd_loss(jittor.nn.Module):
    """
    Earth Mover Distance loss
    """
    def __init__(self, dist_r=2,
        use_l1loss=True, l1loss_coef=0.0):
        super(emd_loss, self).__init__()
        self.dist_r = dist_r
        self.use_l1loss = use_l1loss
        self.l1loss_coef = l1loss_coef

    def check_type_forward(self, in_types):
        assert len(in_types) == 2

        x_type, y_type = in_types
        assert x_type.size()[0] == y_type.shape[0]
        assert x_type.size()[0] > 0
        assert x_type.ndim == 2
        assert y_type.ndim == 2

    def execute(self, x, y_true):
        self.check_type_forward((x, y_true))

        cdf_x = jittor.cumsum(x, dim=-1)
        cdf_ytrue = jittor.cumsum(y_true, dim=-1)
        if self.dist_r == 2:
            samplewise_emd = jittor.sqrt(jittor.mean(jittor.pow(cdf_ytrue - cdf_x, 2), dim=-1))
        else:
            samplewise_emd = jittor.mean(jittor.abs(cdf_ytrue - cdf_x), dim=-1)
        loss = jittor.mean(samplewise_emd)
        if self.use_l1loss:
            rate_scale =  jittor.array([float(i+1) for i in range(x.size()[1])], dtype=x.dtype)
            x_mean = jittor.mean(x * rate_scale, dim=-1)
            y_true_mean = jittor.mean(y_true * rate_scale, dim=-1)
            l1loss = jittor.mean(jittor.abs(x_mean - y_true_mean))
            loss += l1loss * self.l1loss_coef
        return loss

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def __str__(self):
        fmtstr = '{name}:{avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print(' - '.join(entries))#, flush=True)
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def load_checkpoint(checkpoint_fpath, model, optimizer, strict=False, parallel_saved=True):
    t = time.time()
    checkpoint = jittor.load(checkpoint_fpath)
    if parallel_saved:
        new_state_dict = checkpoint['state_dict']
        print('  -> load %s' % checkpoint_fpath)
    else:
        state_dict = checkpoint['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.find('module')>=0:
                name = k[7:] # remove `module.` for parallel saved model
            else:
                name = k
                print(k,name)
            new_state_dict[name] = v
        print('  -> load %s (remove module)' % checkpoint_fpath)

    model.load_state_dict(new_state_dict)
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])            
#     print(f'   %Time: {float((time.time()-t)/60):.2f}min -- loading checkpoint')
    return model, optimizer, checkpoint['epoch']

def emd_dis(x, y_true, dist_r = 1):
    cdf_x = jittor.cumsum(x, dim=-1)
    cdf_ytrue = jittor.cumsum(y_true, dim=-1)
    if dist_r == 2:
        samplewise_emd = jittor.sqrt(jittor.mean(jittor.pow(cdf_ytrue - cdf_x, 2), dim=-1))
    else:
        samplewise_emd = jittor.mean(jittor.abs(cdf_ytrue - cdf_x), dim=-1)
    loss = jittor.mean(samplewise_emd)
    return loss

def get_lr(optimizer):
    try:
        for param_group in optimizer.param_groups:
            print('param_group',param_group['lr'])
            return param_group['lr']
    except Exception as e:
        print(e)
        return None

def binary_accuracy(y_pred, input_label, bins=10):
    rate_scale = jittor.array([float(i+1) for i in range(bins)], dtype=y_pred.dtype)
    threshold  = float(bins / 2)
    _pred = jittor.sum(y_pred * rate_scale, dim=-1)
    _label = jittor.sum(input_label * rate_scale, dim=-1)
    diff = (((_pred-threshold) * (_label-threshold)) >= 0)
    acc = jittor.sum(diff.float()) / _pred.numel()
    return acc

def cal_metrics(output, target, bins=10):
    output = np.concatenate(output)
    target = np.concatenate(target)
    scores_mean = np.dot(output, np.arange(1, bins+1))
    labels_mean = np.dot(target, np.arange(1, bins+1))
    srcc, _ = spearmanr(scores_mean, labels_mean)
    plcc, _ = pearsonr(scores_mean, labels_mean)      
    mse = ((scores_mean - labels_mean)**2).mean(axis=None)
    diff = (((scores_mean-float(bins/2)) * (labels_mean-float(bins/2))) >= 0)
    acc = np.sum(diff) / len(scores_mean) * 100
    output_tensor = jittor.array(output)
    target_tensor = jittor.array(target)
    with jittor.no_grad():
        emd1 = emd_dis(output_tensor, target_tensor, dist_r = 1)
        emd2 = emd_dis(output_tensor, target_tensor, dist_r = 2)             
        return [mse, srcc, plcc, acc, emd1, emd2]