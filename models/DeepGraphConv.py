from os.path import join
from collections import OrderedDict

import pdb
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential as Seq
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GCNConv, GraphConv, GatedGraphConv, GATConv, SGConv, GINConv, GENConv, DeepGCNLayer
from torch_geometric.nn import GraphConv, TopKPooling, SAGPooling
from torch_geometric.nn import global_mean_pool as gavgp, global_max_pool as gmp, global_add_pool as gap
from torch_geometric.transforms.normalize_features import NormalizeFeatures

class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class DeepGraphConv(torch.nn.Module):
    def __init__(self, edge_agg='latent', resample=0, input_dim=1024, hidden_dim=256, 
        linear_dim=256, use_edges=False, dropout=0.25, n_classes=4):
        super(DeepGraphConv, self).__init__()
        self.use_edges = use_edges
        self.resample = resample
        self.edge_agg = edge_agg
        self.pool = False    # can be  reviewed
        
        if self.resample > 0:
            self.fc = nn.Sequential(*[nn.Dropout(self.resample)])

        self.conv1 = GINConv(Seq(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv2 = GINConv(Seq(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv3 = GINConv(Seq(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        
        self.path_attention_head = Attn_Net_Gated(L=hidden_dim, D=hidden_dim, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.classifier = torch.nn.Linear(hidden_dim, n_classes)

    def relocate(self):
        from torch_geometric.nn import DataParallel
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.conv1 = nn.DataParallel(self.conv1, device_ids=device_ids).to('cuda:0')
            self.conv2 = nn.DataParallel(self.conv2, device_ids=device_ids).to('cuda:0')
            self.conv3 = nn.DataParallel(self.conv3, device_ids=device_ids).to('cuda:0')
            self.path_attention_head = nn.DataParallel(self.path_attention_head, device_ids=device_ids).to('cuda:0')

        self.path_rho = self.path_rho.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, **kwargs):
        data = kwargs['wsi']
        x = data.x
        if self.edge_agg == 'spatial':
            edge_index = data.edge_index
        elif self.edge_agg == 'latent':
            edge_index = data.edge_latent
        batch = data.batch
        edge_attr = None

        if self.resample:
            x = self.fc(x)

        x1 = F.relu(self.conv1(x, edge_index, edge_attr))
        if self.pool:
            x1, edge_index, _, batch, perm, score = self.pool1(x1, edge_index, None, batch)
            x1_cat = torch.cat([gmp(x1, batch), gap(x1, batch)], dim=1)

        x2 = F.relu(self.conv2(x1, edge_index, edge_attr))
        if self.pool:
            x2, edge_index, _, batch, perm, score = self.pool2(x2, edge_index, None, batch)
            x2_cat = torch.cat([gmp(x2, batch), gap(x2, batch)], dim=1)

        x3 = F.relu(self.conv3(x2, edge_index, edge_attr))
        h_path = x3

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1) , h_path)
        h_path = self.path_rho(h_path).squeeze()
        h = h_path # [256] vector
        
        logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        return logits, hazards, S

if __name__ == '__main__':
    x = torch.load('/data115_2/jsh/LEOPARD_GCN/Leopard_1024/CTrans/pt_files/case_radboud_0000.pt').to('cuda')
    model = DeepGraphConv(input_dim=768).to('cuda')
    logits, hazards, S= model(wsi=x)
    print(hazards.shape)