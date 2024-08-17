import torch
from pathlib import Path
import json
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import numpy as np
import os
import logging

def make_weights_for_balanced_classes_split(dataset):
    
    N = float(len(dataset))                                               
    weight_per_surv_class = [N/len(dataset.slide_surv_ids[c]) for c in range(len(dataset.slide_surv_ids))]                                                                                                  
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        (surv_y) = dataset.getlabel(idx)                        
        weight[idx] = weight_per_surv_class[surv_y]                            
    return torch.DoubleTensor(weight)

#---->read yaml
import yaml
from addict import Dict
def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)

def get_logger(name, verbosity=2):
    log_levels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG
    }
    msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, log_levels.keys())
    assert verbosity in log_levels, msg_verbosity
    logger = logging.getLogger(name)
    logger.setLevel(log_levels[verbosity])
    return logger


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=2, patience=5, stop_epoch=20, verbose=False, logger=None, multi_gpus=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.logger = logger
        self.multi_gpus = multi_gpus

    def __call__(self, epoch, metric, models, ckpt_name = 'checkpoint.pt'):

        score = metric

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric, models, ckpt_name, epoch)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.save_checkpoint(metric, models, ckpt_name, epoch)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, metric, model, ckpt_name, epoch):
        '''Saves model when validation loss decrease.'''
        self.logger.info(f'metric increase ({self.best_score:.6f} --> {metric:.6f}).  Saving model ... at epoch {epoch}')  
        if self.multi_gpus == 'DataParallel':
            torch.save(model.module.state_dict(), os.path.join(ckpt_name, 'model.pt'))
        else:
            torch.save(model.state_dict(), os.path.join(ckpt_name, 'model.pt'))

def normalize(a, b):
    # return a/(a+b), b/(a+b)
    # 将两个数放入数组中
    arr = np.array([a, b])
    
    # 使用 softmax 函数计算归一化后的概率分布
    softmax_result = np.exp(arr) / np.sum(np.exp(arr))
    
    # 返回归一化后的结果
    return 5 * softmax_result[0], 5 * softmax_result[1]