import argparse
import importlib
import inspect
import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset
import yaml
from Dataset.dataloader import LEDataLoader
from utils.tools import read_yaml
from addict import Dict
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler, SequentialSampler
from tqdm import tqdm
from utils.metric import c_index

cpu_num = 8
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)

def collate_MT(batch):

    for item in batch:
        if torch.is_tensor(item[0]):
            img = torch.cat([item[0] for item in batch], dim = 0)
        else:
            img = item[0]
    
    case_id = np.array([item[1] for item in batch])
    return [img, case_id]

def init_model(cfg):
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

    state_dict_path = cfg.Data['model_path']
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    
    return model

class LEDataset_Infer(Dataset):
        def __init__(self, 
            csv_path = 'label.csv', data_dir=None):
            super(LEDataset_Infer, self).__init__()

            self.slide_data = pd.read_csv(csv_path, low_memory=False)
            self.data_dir = data_dir

        def __getitem__(self, idx):
            case_id = self.slide_data['case_id'][idx].split('.tif')[0]
            data_dir = self.data_dir
            wsi_path  = os.path.join(data_dir, 'pt_files', '{}'.format(case_id)+'.pt')   
            path_features = torch.load(wsi_path)
            return (path_features, case_id)
        
        def __len__(self):
            return len(self.slide_data)

def main(args, cfg):
    model = init_model(cfg)
    model.to(args.device)
    model.eval()

    dataset = LEDataset_Infer(args.csv_path, args.data_dir)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler = RandomSampler(dataset), collate_fn=collate_MT)

    case_id_list = []
    risk_list = []

    with torch.no_grad():
        for batch_idx, (path_features, case_id) in enumerate(tqdm(data_loader)):
            path_features = path_features.to(args.device)
            if cfg.Model.name == 'DSMIL':
                max_prediction, bag_prediction, hazards_i, S_i, hazards_b, S_b = model(wsi=path_features)
                surv_logits = 0.5*torch.sigmoid(max_prediction)+0.5*torch.sigmoid(bag_prediction)
                hazards = torch.sigmoid(surv_logits)
                S = torch.cumprod(1 - surv_logits, dim=1)
            else:
                surv_logits, hazards, S = model(wsi=path_features)
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()

            case_id_list.append(case_id)
            risk_list.append(risk)
            data_dict = {'case_id':case_id_list, 'risk_list':risk_list}
            csv_path = os.path.join('temp_test/results_temp', 'inference_valid_2.csv')
            df = pd.DataFrame(data_dict)
            df.to_csv(csv_path, index=False)

        ## debug
        csv_path='/data115_2/jsh/LEOPARD/leopard_labels/training_labels.csv'
        data_dir = '/data115_2/LEOPARD/FEATURES/Leopard_256_at_0.25mpp/'
        split_dir = '/data115_2/jsh/LEOPARD/splits'
        # folds = 1
        all_data_loader = LEDataLoader(csv_path, data_dir, split_dir, fold=2)
        train_data_loader, valid_data_loader = all_data_loader.get_dataloader()
        data_loader = train_data_loader
        data_loader = valid_data_loader

        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))

        # for batch_idx, (path_features, Y_surv, event_time, c, case_id) in enumerate(tqdm(data_loader)):
        
        #     path_features = path_features.to(args.device)
        #     Y_surv = Y_surv.type(torch.LongTensor).to(args.device)
        #     c = 1.0 - c.type(torch.FloatTensor).to(args.device) # censor+event=1
        #     with torch.no_grad():   
        #         if cfg.Model.name == 'DSMIL':
        #             max_prediction, bag_prediction, hazards_i, S_i, hazards_b, S_b = model(wsi=path_features, label=Y_surv)
        #             logits = 0.5*torch.sigmoid(max_prediction)+0.5*torch.sigmoid(bag_prediction)
        #             S = torch.cumprod(1 - logits, dim=1)
        #         else:
        #             surv_logits, hazards, S = model(wsi = path_features, label=Y_surv)


        #     risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        #     all_risk_scores[batch_idx] = risk
        #     all_censorships[batch_idx] = c.item()
        #     all_event_times[batch_idx] = event_time

        #     case_id_list.append(case_id)
        #     risk_list.append(risk)

        #     ## 前进一格 0046 [-1.858105] ，0403[-1.8509387]，0400 [-1.8561033]
        #     ##         0046[-2.191603]，0403[-2.0884788]，0400[-2.166253]
        #     csv_path = os.path.join('temp_test/results_temp', 'inference_valid_2.csv')
        #     data_dict = {'case_id':case_id_list, 'risk_list':risk_list}
        #     df = pd.DataFrame(data_dict)
        #     df.to_csv(csv_path, index=False)

        # cindex = c_index(all_censorships, all_event_times, all_risk_scores)
        # print("\n\n\n====\033[1;32mValid\033[0m Statistics====")
        # print('\033[1;34mC-index\033[0m: \033[1;31m{:.4f}\033[0m'.format(cindex))
        ##


if __name__ == '__main__':

    args = argparse.ArgumentParser(description='PyTorch Template')
    
    args.add_argument('--save_dir',type=str, default='/data115_2/jsh/LEOPARD/results', help='')
    args.add_argument('--csv_path',type=str, default='temp_test/training_labels.csv', help='')
    args.add_argument('--config',type=str, default='LEOPARD/config/TransMIL.yaml', help='')
    args.add_argument('--data_dir',type=str, default='/data115_2/LEOPARD/FEATURES/Leopard_256_at_0.25mpp/', help='')
    args.add_argument('--batch_size',type=int, default=1, help='')
    args.add_argument('--device',type=str, default='cuda', help='')
    args.add_argument('--multi_gpu_mode',type=str, default='DataParallel', help='Gpus')
    args = args.parse_args()
    
    args.config = 'LEOPARD/config/Patch_GCN_v2.yaml'

    cfg = read_yaml(args.config)
    args.data_dir = cfg.Data.data_dir
    args.save_dir = cfg.Data.save_dir
    args.model_path = cfg.Data.model_path

    if cfg.Data.feature_kind == 'res50':
        args.data_dir = cfg.Data.data_dir
    else:
        args.data_dir = os.path.join(cfg.Data.data_dir, cfg.Data.feature_kind)

    main(args, cfg)

# CUDA_VISIBLE_DEVICES="6,7" python LEOPARD/inference.py
