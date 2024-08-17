import os
import sys
sys.path.append('/data115_2/jsh/LEOPARD')
import numpy as np
import torch
from Dataset.dataset import LEDataset
from utils.tools import *
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler, SequentialSampler
from utils.tools import make_weights_for_balanced_classes_split
from utils.collate import collate_MT
from torch_geometric.data import DataListLoader

class LEDataLoader(DataLoader):
    def __init__(self, csv_path, data_dir, data_dir_multi_scale, split_dir, fold, model_graph=True, shuffle=True,
                         num_workers=4 , use_h5=False, multi_scale=False, random_choice=False,return_list=False,
                         if_three_dataset=False, GAT_graph=False, batch_size=1):
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.data_dir_multi_scale = data_dir_multi_scale
        self.num_workers = num_workers
        self.use_h5 = use_h5
        self.multi_scale = multi_scale
        self.random_choice = random_choice
        self.if_three_dataset = if_three_dataset
        self.batch_size = batch_size
        self.dataset = LEDataset(csv_path = csv_path,
										   data_dir=data_dir,
                                           data_dir_multi=data_dir_multi_scale,
										   shuffle=False, 
										   n_bins=4,
                                           model_graph=model_graph,
										   label_col = 'follow_up_years',
                                           use_h5=self.use_h5,
                                           multi_scale=multi_scale,
                                           random_choice = random_choice,
                                           return_list=return_list,
                                           if_three_dataset=if_three_dataset,  # 是否有train val test
                                           GAT_graph=GAT_graph,
                                           )
        if if_three_dataset:
            train_dataset, val_dataset, test_dataset = self.dataset.return_splits(from_id=False, csv_path='{}/split{}.csv'.format(split_dir, fold))
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.test_dataset = test_dataset
        else:
            train_dataset, val_dataset = self.dataset.return_splits(from_id=False, csv_path='{}/split{}.csv'.format(split_dir, fold))
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

    def get_split_loader(self, split_dataset, is_grapth=False, training=True, weighted=True, batch_size=1):
        """
            return either the validation loader or training loader 
        """
        if is_grapth:
            collate = collate_MT # maybe collate_MT_Graph
        else:
            collate = collate_MT
        
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {'num_workers': self.num_workers} if device.type == "cuda" else {}

        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                if batch_size!=1:   # 在直接回归risk值的方法中, GAT和GAT_custom
                    loader = DataListLoader(split_dataset, batch_sampler=WeightedRandomSampler(weights, len(weights)), num_workers=8, pin_memory=True)
                else:
                    loader = DataLoader(split_dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate, **kwargs)    
            else:
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate, **kwargs)
        else:   # valiadition
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate, **kwargs)

        return loader

    def get_dataloader(self):
        train_dataloader = self.get_split_loader(self.train_dataset, training=True, weighted=True, batch_size=self.batch_size)
        val_dataloader = self.get_split_loader(self.val_dataset, training=False, weighted=False)
        if self.if_three_dataset:
            test_dataloader = self.get_split_loader(self.test_dataset, training=False, weighted=False)
            return train_dataloader, val_dataloader, test_dataloader
        return train_dataloader, val_dataloader

if __name__ == '__main__':
    # csv_path='/data115_2/jsh/LEOPARD/leopard_labels/training_labels.csv'
    # data_dir = '/data115_2/jsh/LEOPARD_GCN/Leopard_512_at_0.25mpp/component_features'
    split_dir = '/data115_2/jsh/LEOPARD/splits'
    folds = 1

    ##  1
    # data_dir = '/data115_2/jsh/LEOPARD_GCN/Leopard_512_at_0.25mpp/CTrans'
    # csv_path = '/data115_2/jsh/LEOPARD_GCN/Leopard_512_at_0.25mpp/CTrans/pt_files/training_labels.csv'
    # for i in range(folds):
    #     data_loader = LEDataLoader(csv_path, data_dir, split_dir, model_graph=True, num_workers=0, fold=i)
    #     train_dataloader, val_dataloader = data_loader.get_dataloader()
    #     for (path_features, Y_surv, event_time, c, case_id) in val_dataloader:
    #         # print(path_features.shape)
    #         # print(Y_surv)
    #         # break
    #         print(case_id)

    ## 2
    # data_dir = '/data115_2/jsh/LEOPARD_GCN/Leopard_512_at_0.25mpp/CTrans_3000'
    # csv_path = '/data115_2/jsh/LEOPARD_GCN/Leopard_512_at_0.25mpp/CTrans_3000/h5_files/training_labels.csv'
    # for i in range(folds):
    #     data_loader = LEDataLoader(csv_path, data_dir, split_dir, model_graph=False, num_workers=0, fold=i, use_h5=True)
    #     train_dataloader, val_dataloader = data_loader.get_dataloader()
    #     for (path_features, Y_surv, event_time, c, case_id) in val_dataloader:
    #         # print(path_features.shape)
    #         # print(Y_surv)
    #         # break
    #         print(case_id)

    ## 3
    # data_dir = '/data115_2/LEOPARD/FEATURES/Leopard_512_at_0.25mpp/CTrans'
    # data_dir_multi_scale = '/data115_2/jsh/LEOPARD_features_2048_at_0.25mpp/CTranspath'
    # csv_path = '/data115_2/jsh/LEOPARD/leopard_labels/training_labels.csv'
    # for i in range(folds):
    #     data_loader = LEDataLoader(csv_path, data_dir, data_dir_multi_scale, split_dir, model_graph=False, num_workers=0, fold=i, use_h5=False, multi_scale=True)
    #     train_dataloader, val_dataloader = data_loader.get_dataloader()
    #     for (path_features, Y_surv, event_time, c, case_id) in val_dataloader:
    #         # print(path_features.shape)
    #         # print(Y_surv)
    #         # break
    #         print()
    #         print(case_id)

    ## 4
    split_dir = '/data115_2/jsh/LEOPARD/split_5fold_train_val_test'
    data_dir = '/data115_2/jsh/LEOPARD_features_512with2048/CTrans'
    csv_path = '/data115_2/jsh/LEOPARD/leopard_labels/training_labels.csv'
    for i in range(folds):
        data_loader = LEDataLoader(csv_path, data_dir, None, split_dir, model_graph=False, num_workers=0, fold=i, use_h5=True, multi_scale=False, if_three_dataset=True)
        train_dataloader, val_dataloader, test_dataloader = data_loader.get_dataloader()
        for (path_features, Y_surv, event_time, c, case_id) in test_dataloader:
            # print(path_features.shape)
            # print(Y_surv)
            # break
            print()
            print(case_id)