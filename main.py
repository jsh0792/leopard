import torch
import os
import argparse
import torch.optim as optim
from utils.tools import read_yaml
from Dataset.dataloader import LEDataLoader
from Trainer.trainer import Trainer
from Trainer.trainer_v2 import Trainer_v2
from Trainer.trainer_v3 import Trainer_v3
import importlib
import inspect
from inspect import signature

cpu_num = 8
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)


def init_model(args, cfg):
    name = cfg.Model['name']
    try:
        print(name)
        Model = getattr(importlib.import_module(f'models.{name}'), name)
    except:
        raise ValueError('Invalid Module File Name or Invalid Class Name!')
    class_args = inspect.getfullargspec(Model.__init__).args[1:]    # ['i_classifier', 'b_classifier']

    args_dict = {}
    for _args in class_args:
        if _args in cfg.Model.keys():
            args_dict[_args] = cfg.Model[_args]
    model = Model(**args_dict)
    if args.multi_gpu_mode == 'DataParallel':
        model = torch.nn.DataParallel(model)
    return model


def main(args, cfg):
    print('dealing ' + str(args.start_fold) + '-' + str(args.end_fold))
    for i in range(args.start_fold, args.end_fold):
        # if cfg.Model['name'] == 'GAT' or cfg.Model['name'] == 'GAT_custom':
        data_loader = LEDataLoader(args.csv_path, args.data_dir, args.data_dir_multi_scale, args.split_dir, model_graph=cfg.Model.model_graph,
                                     fold=i, use_h5=cfg.Data.use_h5, multi_scale=cfg.Data.multi_scale, random_choice=cfg.Data.random_choice,
                                     return_list=cfg.Data.return_list, if_three_dataset=cfg.Data.if_three_dataset, GAT_graph=cfg.Data.GAT_graph,
                                     batch_size=1)
        train_dataloader, val_dataloader = data_loader.get_dataloader()
        model = init_model(args, cfg)
        model.to(args.device)
        if cfg.Optimizer.opt == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=cfg.Optimizer.lr, weight_decay=cfg.Optimizer.weight_decay)
        elif cfg.Optimizer.opt == 'sgd':
            optimizer =  optim.SGD(model.parameters(), lr=cfg.Optimizer.lr, momentum=cfg.Optimizer.momentum)
        else:
            optimizer = optim.RAdam(model.parameters(), lr=cfg.Optimizer.lr, weight_decay=cfg.Optimizer.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.coslr_T)

        if cfg.Data.return_list:
            trainer = Trainer_v2(model, optimizer, args, cfg, args.device, i, train_dataloader, val_dataloader, lr_scheduler=lr_scheduler)      # type=list
        elif cfg.Data.GAT_graph:
            trainer = Trainer_v3(model, optimizer, args, cfg, args.device, i, train_dataloader, val_dataloader, lr_scheduler=lr_scheduler)      # type=list
        else:
            trainer = Trainer(model, optimizer, args, cfg, args.device, i, train_dataloader, val_dataloader, lr_scheduler=lr_scheduler)

        if args.test_phase:
            trainer.test(i)
        else:
            trainer.train(i)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    
    args.add_argument('--save_dir',type=str, default='/data115_2/jsh/LEOPARD/results', help='')
    args.add_argument('--epochs',type=int, default=30, help='')
    args.add_argument('--folds',type=int, default=5, help='')
    args.add_argument('--start_fold',type=int, default=0, help='')
    args.add_argument('--end_fold',type=int, default=5, help='')
    args.add_argument('--csv_path',type=str, default='/data115_2/jsh/LEOPARD/leopard_labels/training_labels.csv', help='')
    args.add_argument('--data_dir',type=str, default='/data115_2/LEOPARD/FEATURES/Leopard_256_at_0.25mpp/', help='')
    args.add_argument('--data_dir_multi_scale',type=str, default='/data115_2/LEOPARD/FEATURES/Leopard_256_at_0.25mpp/', help='')
    args.add_argument('--split_dir',type=str, default='/data115_2/jsh/LEOPARD/splits', help='')
    args.add_argument('--optimizer',type=str, default='adam', help='')
    args.add_argument('--lr',type=float, default=0.0001, help='')
    args.add_argument('--weight_decay',type=float, default=0.000005, help='学习率衰减参数')
    args.add_argument('--momentum',type=float, default=0.4, help='sgd动量')
    args.add_argument('--coslr_T', type=int, default=10, help='CosineAnnealingLR的T_max参数')
    args.add_argument('--lr_scheduler', action='store_true', default=False, help='是否使用lr_scheduler')
    args.add_argument('--config', type=str, help='yaml file')
    args.add_argument('--batch_size',type=int, default=1, help='')
    args.add_argument('--device',type=str, default='cuda', help='')
    args.add_argument('--multi_gpu_mode',type=str, default='DataParallel', help='Gpus')
    args.add_argument('--test_phase', action='store_true', default=False, help='test')

    args = args.parse_args()
    cfg = read_yaml(args.config)

    args.csv_path = cfg.Data.csv_path

    if cfg.Model.model_graph:
        args.save_dir = os.path.join(cfg.Data.save_dir, cfg.Model.name, 'mpdel_graph', cfg.Data.feature_kind)
    else:
        args.save_dir = os.path.join(cfg.Data.save_dir, cfg.Model.name, cfg.Data.feature_kind)

    os.makedirs(args.save_dir, exist_ok=True)   # mkdir创建一级目录 makedirs()


    if cfg.Data.feature_kind == 'UNI' or cfg.Data.feature_kind == 'CTrans':
        args.data_dir = os.path.join(cfg.Data.data_dir, cfg.Data.feature_kind)
        args.data_dir_multi_scale = os.path.join(cfg.Data.data_dir_multi_scale, cfg.Data.feature_kind)    
        print(args.data_dir)
        print(args.data_dir_multi_scale)
    else:
        args.data_dir = cfg.Data.data_dir
        args.data_dir_multi_scale = cfg.Data.data_dir_multi_scale

    if 'split_dir' in cfg.Data:     # need to review when testing:
        args.split_dir = cfg.Data.split_dir

    print(args.split_dir)
    print(args.data_dir)
    print(args.data_dir_multi_scale)
    main(args, cfg)