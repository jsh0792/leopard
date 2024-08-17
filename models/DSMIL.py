import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import importlib
import inspect
import sys
sys.path.append('/data115_2/jsh/LEOPARD')

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats):
        x = self.fc(feats)
        return feats, x

class BClassifier(nn.Module):
    # nonlinear=False
    def __init__(self, input_size, output_class, dropout_v=0.0, dim = 128, nonlinear=False, passing_v=True, max_k = 1, matrixversion=False): # K, L, N
        super(BClassifier, self).__init__()
        
        # if nonlinear:
        #     self.q = nn.Sequential(nn.Linear(input_size, dim), nn.ReLU(), nn.Linear(dim, dim), nn.Tanh())
        # else:

        self.q = nn.Linear(input_size, dim)
        
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        self.matrixversion = matrixversion
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)  
        self.max_k = max_k
        self.output_class = output_class
        self.proj = nn.Linear(input_size, input_size, bias=True)
        
    def forward(self, feats, c): # N x K, N x Cls
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted

        Q = F.normalize(Q, dim=-1, p=2)
        V = F.normalize(V, dim=-1, p=2)

        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0:self.max_k, :].squeeze()) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        # A = F.softmax( F.normalize(A, dim=-1, p=2), 1)
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        B = self.proj(B)
        if self.matrixversion:
            B = F.gelu(B)
            return B, A
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B 
    
class DSMIL(nn.Module):
    def __init__(self, n_classes, input_dim):
        super(DSMIL, self).__init__()
        i_classifier = FCLayer(input_dim, n_classes)
        b_classifier = BClassifier(input_dim, n_classes)
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, **kwargs):
        x = kwargs['wsi']
        feats, classes = self.i_classifier(x)   # [num_patch, 1024], [num_patch, num_class]
        prediction_bag, A, B = self.b_classifier(feats, classes)

        ## method1
        max_prediction, _ = torch.max(classes, 0)
        max_prediction = max_prediction.unsqueeze(0)
        hazards_i = torch.sigmoid(max_prediction)
        S_i = torch.cumprod(1 - hazards_i, dim=1)

        ## method2
        # hazards_i = torch.sigmoid(classes)
        # S_i = torch.cumprod(1 - hazards_i, dim=1)

        hazards_b = torch.sigmoid(prediction_bag)
        S_b = torch.cumprod(1 - hazards_b, dim=1)
        
        return max_prediction, prediction_bag, hazards_i, S_i, hazards_b, S_b

if __name__ == "__main__":
    data = torch.randn((6000, 512)).cuda()

    # i_classifier = FCLayer(in_size=1024, out_size=4)
    # embed_dim = 1024
    # embed_num = 256
    # b_classifier = BClassifier(input_size=1024, output_class=4, nonlinear=1, max_k=1, matrixversion=False)
    
    # method0
    model = DSMIL(n_classes=4, input_dim=512).cuda()
    print(model.eval())
    ins_prediction, bag_prediction, hazards_i, S_i, hazards_b, S_b = model(wsi = data)
    print(ins_prediction, bag_prediction)
    print('DSMIL:')
    print(hazards_i.shape)
    print(S_i)
    print(hazards_b.shape)
    print(S_b)

    # method1
    ## `object = module.name(a, b=1)`
    # dict = {'i_classifier':i_classifier, 'b_classifier':b_classifier}
    # import DSMIL as module_data
    # model = module_data.DSMIL(**dict).cuda()
    # print(model.eval())
    # ins_prediction, bag_prediction, hazards_i, S_i, hazards_b, S_b = model(wsi = data)
    # print(ins_prediction, bag_prediction)