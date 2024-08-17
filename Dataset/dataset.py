import numpy as np
import pandas as pd
import torch
import os
import sys
sys.path.append('/data115_2/jsh')
from torch.utils.data import Dataset
from pathlib import Path
import glob
from Patch_GCN.datasets.BatchWSI import BatchWSI
import h5py

class LEDataset(Dataset):
    def __init__(self,
        # data_dir = '/data1/ganzy/data_gzy/pt_files/BRCA'
        csv_path = 'label.csv', data_dir=None, data_dir_multi=None, shuffle = False, seed = 7, n_bins = 4,
                     label_col = None, eps=1e-6, model_graph=False, use_h5=False, multi_scale=False, random_choice=False, return_list=False,
                     if_three_dataset=False, GAT_graph=False):
        super(LEDataset, self).__init__()

        self.seed = seed
        self.data_dir = data_dir    
        self.data_dir_multi = data_dir_multi    # 多尺度特征路径
        self.model_graph = model_graph
        self.use_h5 = use_h5
        self.multi_scale = multi_scale          # 是否多尺度特征
        self.random_choice = random_choice      # 选择部分patch
        self.return_list = return_list
        self.if_three_dataset = if_three_dataset
        self.GAT_graph = GAT_graph

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        slide_data = pd.read_csv(csv_path, low_memory=False)
        patients_df = slide_data.copy()


        patient_dict = {}

        if use_h5 == True:
            file_names = os.listdir(os.path.join(data_dir, 'h5_files')) 
        else:
            file_names = os.listdir(os.path.join(data_dir, 'pt_files')) 

        for file_name in file_names:
            if 'fullname' in slide_data.columns:    # 每个病人有多张WSI
                patient = file_name[:-3]    # case_radboud_0596_0.pt -> case_radboud_0596_0
            else:
                patient = file_name[:17]    # case_radboud_0596_0.pt -> case_radboud_0596 or case_radboud_0596.pt -> case_radboud_0596

            if patient in patient_dict:
                patient_dict[patient].append(file_name)
            else:
                patient_dict[patient] = [file_name]
        self.patient_dict = patient_dict    # {'case_radboud_0463': ['case_radboud_0463.pt'], 'case_radboud_0325': ['case_radboud_0325.pt'], ...
                                            # {'case_radboud_0574': ['case_radboud_0574_1.pt', 'case_radboud_0574_0.pt', 'case_radboud_0574_2.pt'],...
        #                    case_id  event  follow_up_years
        # 0    case_radboud_0596.tif      0         3.734428
        # 1    case_radboud_0134.tif      0        11.600274

        # uncensored_df = patients_df[patients_df['event'] < 1] # event<1表示 
        uncensored_df = patients_df[patients_df['event'] == 1] # event<1表示 
        label_col = 'follow_up_years'
        self.label_col = label_col
        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps
        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))
        slide_data = patients_df    # 离散化划分了生存时间

        #                    case_id  event  label  follow_up_years
        # 0    case_radboud_0596.tif      0      1         3.734428
        # 1    case_radboud_0134.tif      0      3        11.600274

        label_dict = {}     # dict{}中对应的组成成分中，是label和censorship的组合结果
        key_count = 0
        for i in range(len(q_bins)-1):
            for c in [0, 1]:
                print('{} : {}'.format((i, c), key_count))
                label_dict.update({(i, c):key_count})
                key_count+=1
                self.label_dict = label_dict

        self.label_dict = label_dict
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = (int)(key)
            censorship = slide_data.loc[i, 'event']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_dict[key] 

        #                    case_id  event  label  follow_up_years  disc_label
        # 0    case_radboud_0596.tif      0      2         3.734428         1.0
        # 1    case_radboud_0134.tif      0      6        11.600274         3.0
        self.slide_data = slide_data
        print(slide_data)

    def get_split_from_df(self, all_splits: dict, split_key: str='train', scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['case_id'].isin(split.tolist()) # 根据split文件中的划分，进行mask，分别得到三个MTDataset
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, label_col='follow_up_years',model_graph=self.model_graph, patient_dict=self.patient_dict, data_dir=self.data_dir,
                     data_dir_multi=self.data_dir_multi, use_h5=self.use_h5, multi_scale=self.multi_scale, 
                     random_choice=self.random_choice, return_list=self.return_list, if_three_dataset=self.if_three_dataset,
                     GAT_graph=self.GAT_graph)
        else:
            split = None
        return split

    def return_splits(self, from_id: bool=True, csv_path: str=None):
        if from_id:
            raise NotImplementedError
        else:
            assert csv_path 
            all_splits = pd.read_csv(csv_path)
            train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')
            if self.if_three_dataset:
                test_split = self.get_split_from_df(all_splits=all_splits, split_key='test')
                return train_split, val_split, test_split
        return train_split, val_split

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):

        if 'fullname' in self.slide_data.columns:   # 当WSI按照联通区域进行划分时
            if self.use_h5:
                case_id = self.slide_data['fullname'][idx].split('.h5')[0]
            else:
                case_id = self.slide_data['fullname'][idx].split('.pt')[0]
        else:
            case_id = self.slide_data['case_id'][idx].split('.tif')[0]

        Y_surv = self.slide_data['disc_label'][idx]
        event_time = self.slide_data[self.label_col][idx]
        e = self.slide_data['event'][idx]
        slide_ids = self.patient_dict[case_id]  # list

        path_features = []

        ##
        path_features_another_scale = []
        another_scale_dir = '/data115_2/jsh/LEOPARD_features_2048_at_0.25mpp/CTrans/'
        ##

        for slide_id in slide_ids:
            if self.use_h5:
                wsi_path = os.path.join(self.data_dir, 'h5_files', slide_id)
                with h5py.File(wsi_path, 'r') as file:
                    key = 'features'
                    wsi_bag_tensor = torch.from_numpy(file[key][:]).to(torch.float32)

                    if self.random_choice:
                        N = wsi_bag_tensor.shape[0]
                        num_elements_to_select = int(0.7 * N)
                        random_indices = np.random.choice(N, num_elements_to_select, replace=False)
                        wsi_bag_tensor = wsi_bag_tensor[random_indices]
                    path_features.append(wsi_bag_tensor)
                
                if self.return_list:    # 当以list形式返回两个尺度特征时
                    wsi_path = os.path.join(another_scale_dir, 'h5_files', slide_id)
                    with h5py.File(wsi_path, 'r') as file:
                        key = 'features'
                        wsi_bag_tensor = torch.from_numpy(file[key][:]).to(torch.float32)
                        path_features_another_scale.append(wsi_bag_tensor)

            else:
                wsi_path = os.path.join(self.data_dir, 'pt_files', slide_id)
                wsi_bag = torch.load(wsi_path) 
                import torch_geometric.transforms as T
                from torch_geometric.transforms import Polar
                from torch_geometric.data import Data
                
                if self.GAT_graph:
                    polar_transform = Polar()
                    transfer = T.ToSparseTensor()
                    data_re = Data(x=wsi_bag.x[:,:768], edge_index=wsi_bag.edge_index)
                    mock_data = Data(x=wsi_bag.x[:,:768], edge_index=wsi_bag.edge_index, pos=wsi_bag.pos)
                    
                    data_re.pos = wsi_bag.pos
                    data_re_polar = polar_transform(mock_data)
                    polar_edge_attr = data_re_polar.edge_attr

                    wsi_bag = transfer(data_re)
                    wsi_bag.edge_attr = polar_edge_attr
                    wsi_bag.pos = wsi_bag.pos

                if self.multi_scale:
                    wsi_path_multi_scale = os.path.join(self.data_dir_multi, 'pt_files', slide_id)
                    wsi_bag_multi_scale = torch.load(wsi_path_multi_scale)

                    ## temp add 构建三个尺度特征
                    data_dir_multi_2 = '/data115_2/jsh/LEOPARD_features_4096_at_0.25mpp/CTrans'
                    wsi_path_multi_scale_2 = os.path.join(data_dir_multi_2, 'pt_files', slide_id)
                    wsi_bag_multi_scale_2 = torch.load(wsi_path_multi_scale_2)
                    ##
                    wsi_bag = torch.concat([wsi_bag, wsi_bag_multi_scale, wsi_bag_multi_scale_2], dim=0)
                path_features.append(wsi_bag)

        if self.model_graph:
            path_features = BatchWSI.from_data_list(path_features, update_cat_dims={'edge_latent': 1})
            return ([path_features], Y_surv, event_time, e, case_id)
        else:
            path_features = torch.cat(path_features, dim=0)
            if self.return_list:
                path_features_another_scale = torch.cat(path_features_another_scale, dim=0)
                return ([path_features, path_features_another_scale], Y_surv, event_time, e, case_id)

            return ( [path_features, path_features], Y_surv, event_time, e, case_id)
    
    def getlabel(self, ids):
        return (self.slide_data['label'][ids])

class Generic_Split(LEDataset):
    def __init__(self, slide_data, label_col='survival_month', model_graph=None, patient_dict=None, data_dir=None, 
                    data_dir_multi=None, use_h5=False, multi_scale=False, random_choice=False, return_list=False,
                    if_three_dataset=False, GAT_graph=False):
        # super().__init__()
        self.slide_data = slide_data
        self.label_col = label_col
        self.data_dir = data_dir
        self.data_dir_multi = data_dir_multi
        self.patient_dict = patient_dict
        self.model_graph = model_graph
        self.use_h5 = use_h5
        self.multi_scale = multi_scale
        self.random_choice = random_choice
        self.return_list = return_list
        self.if_three_dataset = if_three_dataset
        self.GAT_graph = GAT_graph
        
        self.surv_discclass_num = len(self.slide_data['label'].value_counts())
        self.slide_surv_ids = [[] for i in range(self.surv_discclass_num)]
        for i in range(self.surv_discclass_num):
            self.slide_surv_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def __len__(self):
        return len(self.slide_data)


if __name__ == '__main__':
    # 1 
    # data_dir = '/data115_2/LEOPARD/FEATURES/Leopard_512_at_0.25mpp/CTrans'
    # # csv_path = '/data115_2/jsh/temp_test/training_labels_1.csv'
    # csv_path = '/data115_2/jsh/LEOPARD/leopard_labels/training_labels.csv'
    # dataset = LEDataset(csv_path, data_dir)
    # for (path_features, Y_surv, event_time, c, case_id) in dataset:
    #     print(path_features)
    #     print(Y_surv)

    # 2
    # data_dir = '/data115_2/jsh/LEOPARD_GCN/Leopard_512_at_0.25mpp/component_features'
    # csv_path = '/data115_2/jsh/LEOPARD/leopard_labels/training_labels.csv'
    # dataset = LEDataset(csv_path, data_dir, model_graph=True)
    # for (path_features, Y_surv, event_time, c, case_id) in dataset:
    #     print('dealing' + str(case_id) )

    # 3
    # data_dir = '/data115_2/jsh/LEOPARD_GCN/Leopard_512_at_0.25mpp/CTrans'
    # csv_path = '/data115_2/jsh/LEOPARD_GCN/Leopard_512_at_0.25mpp/CTrans/pt_files/training_labels.csv'
    # dataset = LEDataset(csv_path, data_dir, model_graph=True)
    # for (path_features, Y_surv, event_time, c, case_id) in dataset:
    #     print('dealing' + str(case_id) )

    # 4
    # data_dir = '/data115_2/jsh/LEOPARD_GCN/Leopard_512_at_0.25mpp_disrect_nograph/Ctrans'
    # csv_path = '/data115_2/jsh/LEOPARD_GCN/Leopard_512_at_0.25mpp_disrect_nograph/Ctrans/pt_files/training_labels.csv'
    # dataset = LEDataset(csv_path, data_dir, model_graph=False)
    # for (path_features, Y_surv, event_time, c, case_id) in dataset:
    #     print('dealing' + str(case_id) )
    
    # 5
    # data_dir = '/data115_2/jsh/LEOPARD_GCN/Leopard_512_at_0.25mpp/CTrans_3000'
    # csv_path = '/data115_2/jsh/LEOPARD_GCN/Leopard_512_at_0.25mpp/CTrans_3000/h5_files/training_labels.csv'
    # dataset = LEDataset(csv_path, data_dir, model_graph=False, use_h5=True)
    # for (path_features, Y_surv, event_time, c, case_id) in dataset:
    #     print('dealing' + str(case_id) )
    
    # # 6
    # data_dir = '/data115_2/LEOPARD/FEATURES/Leopard_512_at_0.25mpp/CTrans'
    # data_dir_multi_scale = '/data115_2/jsh/LEOPARD_features_2048_at_0.25mpp/CTrans'
    # csv_path = '/data115_2/jsh/LEOPARD/leopard_labels/training_labels.csv'
    # dataset = LEDataset(csv_path, data_dir, data_dir_multi_scale, model_graph=False, use_h5=True, multi_scale=True, random_choice=True)
    # for (path_features, Y_surv, event_time, c, case_id) in dataset:
    #     print()
    #     print('dealing' + str(case_id) )


    # 7
    data_dir = '/data115_2/LEOPARD/FEATURES/Leopard_512_at_0.25mpp/CTrans'
    data_dir_multi_scale = '/data115_2/jsh/LEOPARD_features_2048_at_0.25mpp/CTrans'
    csv_path = '/data115_2/jsh/LEOPARD/leopard_labels/training_labels.csv'
    dataset = LEDataset(csv_path, data_dir, data_dir_multi_scale, model_graph=False, use_h5=True, multi_scale=True, random_choice=True)
    for (path_features, Y_surv, event_time, c, case_id) in dataset:
        print()
        print('dealing' + str(case_id) )