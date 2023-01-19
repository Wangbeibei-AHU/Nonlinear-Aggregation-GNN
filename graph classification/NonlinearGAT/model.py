import time
import copy
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d
from torch_geometric.nn import global_mean_pool, global_add_pool
from layer import GATConv_G, GATConv_P, GATConv_S
from util import Config
import pdb


class NonLinearGAT(nn.Module):
    def __init__(self, config: Config):
        super(NonLinearGAT, self).__init__()
        
        self.nhid = config.nhid
        self.nclass = config.nclass
        self.readout = config.readout
        self.dropout = config.dropout
        
        NonLinear = [True, False]
        self.conv_layers = nn.ModuleList()
        for i in range(config.n_layer):
            if config.mod == 'Generalized-mean':
                self.conv_layers.append(GATConv_G(config, NonLinear[i]))
            if config.mod == 'Polynomial':
                self.conv_layers.append(GATConv_P(config, NonLinear[i]))
            if config.mod == 'Softmax':
                self.conv_layers.append(GATConv_S(config, NonLinear[i]))
        
        self.fc = Linear(config.nfeat, self.nhid, bias=False)
       
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(config.n_layer + 1):
            if layer == 0:
                self.linears_prediction.append(Linear(config.nfeat, self.nhid))
            else:
                self.linears_prediction.append(Linear(self.nhid, self.nhid))
        
        self.bns_fc = torch.nn.ModuleList()
        for layer in range(config.n_layer + 1):
            if layer == 0:
                self.bns_fc.append(BatchNorm1d(config.nfeat))
            else:    
                self.bns_fc.append(BatchNorm1d(self.nhid))
        
        self.linear = Linear(self.nhid, self.nclass)
        
    def forward(self, x, edge_index, batch):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
            
        if self.readout == 'mean':
            output_list = [global_mean_pool(x, batch)]
        else:
            output_list = [global_add_pool(x, batch)] 
        hid_x = self.fc(x)

        for conv in self.conv_layers:
            hid_x = conv(hid_x, edge_index)
            if self.readout == 'mean':
                output_list.append(global_mean_pool(hid_x, batch))               
            else:
                output_list.append(global_add_pool(hid_x, batch))

        score_over_layer = 0
        for layer, h in enumerate(output_list):
            h = self.bns_fc[layer](h)
            score_over_layer += F.relu(self.linears_prediction[layer](h))
            
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        x = self.linear(score_over_layer)
        return F.log_softmax(x, dim=-1)
        
    
    
       
