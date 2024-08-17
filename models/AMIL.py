import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('LEOPARD/models')
from func import *

SIZE_DICT = {'small': [512, 256], 'big': [512, 384]}

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def norm(x):
    norm = torch.norm(x, p=2, dim=1)
    x = x / (norm.expand(1, -1).t() + .0001)
    return x


class AMIL(nn.Module):
    def __init__(self, n_classes=4, input_dim=1024, dropout=0.25, size_arg="small"):
        super(AMIL, self).__init__()
        self.L, self.D = SIZE_DICT[size_arg]
        self.K = 1

        fc = [nn.Linear(input_dim, self.L), nn.ReLU()]

        if dropout:
            fc.append(nn.Dropout(dropout))
        
        self.fc = nn.Sequential(*fc)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Linear(self.L*self.K, n_classes)

        # ----> ood_energy
        self.m_in = -2.5
        self.m_out = -5

        initialize_weights(self)

    def forward(self, **kwargs):
        # NOTE: add an FC before the attention module as is done in CAMEL. Not consistent with the ABMIL paper, which also trains a CNN to extract features, because it is not affordable due to the large amount of instances in WSI.
        # print(kwargs.keys())
        data = kwargs['wsi']

        ## ID
        H = self.fc(data)  # [N, 1024] -> [N, L]  torch.Size([2, 512])

        A = self.attention(H)  # [N, K] K=1  torch.Size([2, 1])

        A = torch.transpose(A, 1, 0)  # [K, N] 1xN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, H)  # [K, L] 1x500
        logits = self.classifier(M) # [K, num_classes] 1xnum_classes
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return logits, hazards, S


if __name__ == "__main__":
    data = torch.randn((2, 768)).cuda()
    model = AMIL(n_classes=1, input_dim=768).cuda()
    print(model.eval())
    logits, hazards, S = model(wsi = data)
    print(logits, hazards, S)