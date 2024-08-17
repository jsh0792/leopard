# -*- coding: utf-8 -*-
import argparse
import sys
sys.path.append('/data115_2/jsh/TEA-graph-master')
# from GAT_dataset import GAT_Dataset

import torch
import torch.nn as nn
from torch_geometric.data import DataListLoader    
from tqdm import tqdm
from torch.nn import LayerNorm
from torch_geometric.nn import global_mean_pool, BatchNorm
from models.Modified_GAT import GATConv as GATConv
from torch_geometric.nn import GraphSizeNorm

from models.model_utils import weight_init
from models.model_utils import decide_loss_type

from models.pre_layer import preprocess
from models.post_layer import postprocess

def Parser_main():
    parser = argparse.ArgumentParser(description="Deep cox analysis model")
    parser.add_argument("--DatasetType", default="TCGA", help="TCGA_BRCA or BORAME or BORAME_Meta or BORAME_Prog",
                        type=str)
    parser.add_argument("--learning_rate", default=0.0001, help="Learning rate", type=float)
    parser.add_argument("--weight_decay", default=0.00005, help="Weight decay rate", type=float)
    parser.add_argument("--clip_grad_norm_value", default=2.0, help="Gradient clipping value", type=float)
    parser.add_argument("--batch_size", default=6, help="batch size", type=int)
    parser.add_argument("--num_epochs", default=50, help="Number of epochs", type=int)
    parser.add_argument("--dropedge_rate", default=0.25, help="Dropedge rate for GAT", type=float)
    parser.add_argument("--dropout_rate", default=0.25, help="Dropout rate for MLP", type=float)
    parser.add_argument("--graph_dropout_rate", default=0.25, help="Node/Edge feature dropout rate", type=float)
    parser.add_argument("--initial_dim", default=100, help="Initial dimension for the GAT", type=int)
    parser.add_argument("--attention_head_num", default=2, help="Number of attention heads for GAT", type=int)
    parser.add_argument("--number_of_layers", default=3, help="Whole number of layer of GAT", type=int)
    parser.add_argument("--FF_number", default=0, help="Selecting set for the five fold cross validation", type=int)
    parser.add_argument("--model", default="GAT_custom", help="GAT_custom/DeepGraphConv/PatchGCN/GIN/MIL/MIL-attention", type=str)
    parser.add_argument("--gpu", default=0, help="Target gpu for calculating loss value", type=int)
    parser.add_argument("--norm_type", default="layer", help="BatchNorm=batch/LayerNorm=layer", type=str)
    parser.add_argument("--MLP_layernum", default=3, help="Number of layers for pre/pose-MLP", type=int)
    parser.add_argument("--with_distance", default="Y", help="Y/N; Including positional information as edge feature", type=str)
    parser.add_argument("--simple_distance", default="N", help="Y/N; Whether multiplying or embedding positional information", type=str)
    parser.add_argument("--loss_type", default="PRELU", help="RELU/Leaky/PRELU", type=str)
    parser.add_argument("--residual_connection", default="Y", help="Y/N", type=str)

    return parser.parse_args()

class GAT_module(torch.nn.Module):

    def __init__(self, input_dim, output_dim, head_num, dropedge_rate, graph_dropout_rate, loss_type, with_edge, simple_distance, norm_type):
        """
        :param input_dim: Input dimension for GAT
        :param output_dim: Output dimension for GAT
        :param head_num: number of heads for GAT
        :param dropedge_rate: Attention-level dropout rate
        :param graph_dropout_rate: Node/Edge feature drop rate
        :param loss_type: Choose the loss type
        :param with_edge: Include the edge feature or not
        :param simple_distance: Simple multiplication of edge feature or not
        :param norm_type: Normalization method
        """

        super(GAT_module, self).__init__()
        self.conv = GATConv([input_dim, input_dim], output_dim, heads=head_num, dropout=dropedge_rate, with_edge=with_edge, simple_distance=simple_distance)
        self.norm_type = norm_type
        if norm_type == "layer":
            self.bn = LayerNorm(output_dim * int(self.conv.heads))
            self.gbn = None
        else:
            self.bn = BatchNorm(output_dim * int(self.conv.heads))
            self.gbn = GraphSizeNorm()
        self.prelu = decide_loss_type(loss_type, output_dim * int(self.conv.heads))
        self.dropout_rate = graph_dropout_rate
        self.with_edge = with_edge

    def reset_parameters(self):

        self.conv.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, x, edge_attr, edge_index, batch):

        if self.training:
            drop_node_mask = x.new_full((x.size(1),), 1 - self.dropout_rate, dtype=torch.float)
            drop_node_mask = torch.bernoulli(drop_node_mask)
            drop_node_mask = torch.reshape(drop_node_mask, (1, drop_node_mask.shape[0]))
            drop_node_feature = x * drop_node_mask

            drop_edge_mask = edge_attr.new_full((edge_attr.size(1),), 1 - self.dropout_rate, dtype=torch.float)
            drop_edge_mask = torch.bernoulli(drop_edge_mask)
            drop_edge_mask = torch.reshape(drop_edge_mask, (1, drop_edge_mask.shape[0]))
            drop_edge_attr = edge_attr * drop_edge_mask
        else:
            drop_node_feature = x
            drop_edge_attr = edge_attr

        if self.with_edge == "Y":
            x_before, attention_value = self.conv((drop_node_feature, drop_node_feature), edge_index,
                                   edge_attr=drop_edge_attr, return_attention_weights=True)
        else:
            x_before, attention_value = self.conv((drop_node_feature, drop_node_feature), edge_index,
                                   edge_attr=None, return_attention_weights=True)
        out_x_temp = 0
        if self.norm_type == "layer":
            for c, item in enumerate(torch.unique(batch)):
                temp = self.bn(x_before[batch == item])
                if c == 0:
                    out_x_temp = temp
                else:
                    out_x_temp = torch.cat((out_x_temp, temp), 0)
        else:
            temp = self.gbn(self.bn(x_before), batch)
            out_x_temp = temp

        x_after = self.prelu(out_x_temp)

        return x_after, attention_value

class GAT_custom(torch.nn.Module):

    def __init__(self,  n_classes=4, input_dim=768, dropout_rate=0.25): # dropout_rate, dropedge_rate, Argument):
        super(GAT_custom, self).__init__()
        torch.manual_seed(12345)

        dim = input_dim
        self.dropout_rate = dropout_rate
        self.dropedge_rate = 0.25
        self.heads_num = 2
        self.include_edge_feature = 'Y'
        self.layer_num = 3
        self.graph_dropout_rate = 0.25
        self.residual = 'Y'
        self.norm_type = 'layer'
        self.input_dim = input_dim

        self.MLP_layernum = 3
        self.loss_type = 'PRELU'
        self.with_distance = 'Y'
        self.simple_distance = 'N'
        self.norm_type = 'layer'
        self.number_of_layers = 3
        self.attention_head_num = 2

        postNum = 0
        self.preprocess = preprocess(attention_head_num=self.attention_head_num, MLP_layernum=self.MLP_layernum, dropout_rate=self.dropout_rate, norm_type=self.norm_type, 
        initial_dim=self.input_dim, with_distance=self.with_distance, simple_distance=self.simple_distance)

        self.conv_list = nn.ModuleList([GAT_module(dim * self.heads_num, dim, self.heads_num, self.dropedge_rate,
                                                   self.graph_dropout_rate, self.loss_type,
                                                   with_edge=self.with_distance,
                                                   simple_distance=self.simple_distance,
                                                   norm_type=self.norm_type) for _ in
                                        range(int(self.number_of_layers))])
        postNum += int(self.heads_num) * len(self.conv_list)

        self.postprocess = postprocess(dim * self.heads_num, self.layer_num, dim * self.heads_num, (self.MLP_layernum-1), dropout_rate)
        self.risk_prediction_layer = nn.Linear(self.postprocess.postlayernum[-1], n_classes)

    def reset_parameters(self):

        self.preprocess.reset_parameters()
        for i in range(int(self.Argument.number_of_layers)):
            self.conv_list[i].reset_parameters()
        self.postprocess.reset_parameters()
        self.risk_prediction_layer.apply(weight_init)

    def forward(self, **kwargs):
        edge_mask=None
        # print(kwargs['wsi'])
        data = kwargs['wsi']

        row, col, _ = data.adj_t.coo()
        preprocessed_input, preprocess_edge_attr = self.preprocess(data, edge_mask)
        batch = data.batch

        x0_glob = global_mean_pool(preprocessed_input, batch)
        x_concat = x0_glob

        x_out = preprocessed_input
        final_x = x_out
        count = 0
        attention_list = []

        for i in range(int(self.layer_num)):
            select_idx = int(i)
            x_temp_out, attention_value = \
                self.conv_list[select_idx](x_out, preprocess_edge_attr, data.adj_t, batch)
            _, _, attention_value = attention_value.coo()
            if len(attention_list) == 0:
                attention_list = torch.reshape(attention_value, (1, attention_value.shape[0], attention_value.shape[1]))
            else:
                attention_list = torch.cat((attention_list, torch.reshape(attention_value, (
                1, attention_value.shape[0], attention_value.shape[1]))), 0)

            x_glob = global_mean_pool(x_temp_out, batch)
            x_concat = torch.cat((x_concat, x_glob), 1)

            if self.residual == "Y":
                x_out = x_temp_out + x_out
            else:
                x_out = x_temp_out

            final_x = x_out
            count = count + 1

        ##
        # postprocessed_output = self.postprocess(x_concat, data.batch)
        # risk = self.risk_prediction_layer(postprocessed_output)

        # if Interpretation_mode:
        #     return risk, final_x, attention_list
        # else:
        #     return risk
        ##

        ## 2
        postprocessed_output = self.postprocess(x_concat, data.batch)
        logits = self.risk_prediction_layer(postprocessed_output)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return logits, hazards, S

# if __name__ == '__main__':
    # Argument = Parser_main()

    # transfer = T.ToSparseTensor()
    # data_re = torch.load('TEA-graph-master/results_temp/case_radboud_0466_0.75_graph_torch_2201.6_artifact_sophis_final.pt')
    # data = transfer(data_re)

    # dataset = GAT_Dataset()
    # dataloader = DataListLoader(dataset,)

    # model = GAT(Argument.dropout_rate, Argument.dropedge_rate, Argument)
    
    # for batch_idx, data in enumerate(dataloader, 1):
    #     out = model(data)
    #     print(out)