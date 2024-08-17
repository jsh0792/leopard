from torch_geometric.data import DataListLoader
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.transforms import Polar
import torch
import torch_geometric.transforms as T

def GAT_Dataset(Dataset):
    def __init__(self):
        super(GAT_Dataset, self).__init__()

    def len(self):
        return 1
    
    def get(self, index):
        data_origin = torch.load('TEA-graph-master/results_temp/case_radboud_0466_0.75_graph_torch_2201.6_artifact_sophis_final.pt')
        transfer = T.ToSparseTensor()
        data_re = Data(x=data_origin.x[:,:1792], edge_index=data_origin.edge_index)
        mock_data = Data(x=data_origin.x[:,:1792], edge_index=data_origin.edge_index, pos=data_origin.pos)
        
        data_re.pos = data_origin.pos
        data_re_polar = self.polar_transform(mock_data)
        polar_edge_attr = data_re_polar.edge_attr

        data = transfer(data_re)
        data.survival = torch.tensor(1)
        data.phase = torch.tensor(1)
        data.mets_class = torch.tensor(1)
        data.stage = torch.tensor(1)
        data.item = 1
        data.edge_attr = polar_edge_attr
        data.pos = data_origin.pos

        return data
