import torch
import torch.nn.functional as F
from torch.nn import Parameter, Sequential, Linear, ReLU, BatchNorm1d, Dropout
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros, ones
from torch_scatter import scatter_add, scatter_mean, scatter
from torch_geometric.utils import add_self_loops, degree
from util import const, MessagePassing
import pdb


class GATConv_G(MessagePassing):
    def __init__(self, config, NonLinear):
        super(GATConv_G, self).__init__('add')
        self.nhid = config.nhid
        self.heads = config.heads
        self.negative_slope = config.alpha
        self.dropout = config.dropout
        self.mod = config.mod
        self.activation = ReLU()
        self.att = Parameter(torch.Tensor(1, self.heads, 2 * self.nhid))
        self.w = Parameter(torch.ones(self.nhid))
        self.l1 = Parameter(torch.FloatTensor(1, self.nhid))
        self.b1 = Parameter(torch.FloatTensor(1, self.nhid))
        self.l2 = Parameter(torch.FloatTensor(self.nhid, self.nhid))
        self.b2 = Parameter(torch.FloatTensor(1, self.nhid))
        
        self.mlp = Sequential(Linear(self.nhid, self.nhid), Dropout(self.dropout), ReLU(), BatchNorm1d(self.nhid), Linear(self.nhid, self.nhid), Dropout(self.dropout), ReLU(), BatchNorm1d(self.nhid))
        self.NonLinear = NonLinear
        if NonLinear:
            self.p = torch.nn.Parameter(torch.ones(1))
            ## or
            # self.p = torch.nn.Parameter(torch.FloatTensor(1))
            # torch.nn.init.uniform_(self.p.data, -1, 1)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)        
        ones(self.l1)
        zeros(self.b1)
        const(self.l2, 1 / self.nhid)
        zeros(self.b2)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = x.view(-1, self.heads, self.nhid)
        output = self.propagate(edge_index, x=x, num_nodes=x.size(0))
        output = self.mlp(output)
        return output

    def message(self, x_i, x_j, edge_index, num_nodes):
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0], None, num_nodes)
        
        if self.NonLinear:
            p = torch.sigmoid(self.p)+1.
            mu = x_j.min()
            x1 = torch.pow(x_j-mu+1e-6, p)
            x2 = x1 * alpha.view(-1, self.heads, 1)
            output = torch.pow(x2+1e-6, 1./p) + mu
        else:
            output = x_j * alpha.view(-1, self.heads, 1)
        return output

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads * self.nhid)
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.nhid, self.heads)

class GATConv_P(MessagePassing):
    def __init__(self, config, NonLinear):
        super(GATConv_P, self).__init__('add')
        self.nhid = config.nhid
        self.heads = config.heads
        self.negative_slope = config.alpha
        self.dropout = config.dropout
        self.mod = config.mod
        self.activation = ReLU()
        self.att = Parameter(torch.Tensor(1, self.heads, 2 * self.nhid))
        self.w = Parameter(torch.ones(self.nhid))
        self.l1 = Parameter(torch.FloatTensor(1, self.nhid))
        self.b1 = Parameter(torch.FloatTensor(1, self.nhid))
        self.l2 = Parameter(torch.FloatTensor(self.nhid, self.nhid))
        self.b2 = Parameter(torch.FloatTensor(1, self.nhid))
        
        self.mlp = Sequential(Linear(self.nhid, self.nhid), Dropout(self.dropout), ReLU(), BatchNorm1d(self.nhid), Linear(self.nhid, self.nhid), Dropout(self.dropout), ReLU(), BatchNorm1d(self.nhid))
        self.NonLinear = NonLinear
        if NonLinear:
            self.p = torch.nn.Parameter(torch.ones(1))
            ## or
            # self.p = torch.nn.Parameter(torch.FloatTensor(1))
            # torch.nn.init.uniform_(self.p.data, -1, 1)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)        
        ones(self.l1)
        zeros(self.b1)
        const(self.l2, 1 / self.nhid)
        zeros(self.b2)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = x.view(-1, self.heads, self.nhid)
        output = self.propagate(edge_index, x=x, num_nodes=x.size(0))
        output = self.mlp(output)
        return output

    def message(self, x_i, x_j, edge_index, num_nodes):
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0], None, num_nodes)
        
        if self.NonLinear:
            p = torch.sigmoid(self.p)*2.
            mu = x_j.min()
            x1 = torch.pow(x_j-mu+1e-6, p+1)
            x2 = torch.pow(x_j-mu+1e-6, p)
            output = x1 * alpha.view(-1, self.heads, 1)/(x2 * alpha.view(-1, self.heads, 1)+1e-6)+ mu
        else:
            output = x_j * alpha.view(-1, self.heads, 1)
        return output

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads * self.nhid)
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.nhid, self.heads)


class GATConv_S(MessagePassing):
    def __init__(self, config, NonLinear):
        super(GATConv_S, self).__init__('add')
        self.nhid = config.nhid
        self.heads = config.heads
        self.negative_slope = config.alpha
        self.dropout = config.dropout
        self.mod = config.mod
        self.activation = ReLU()
        self.att = Parameter(torch.Tensor(1, self.heads, 2 * self.nhid))
        self.w = Parameter(torch.ones(self.nhid))
        self.l1 = Parameter(torch.FloatTensor(1, self.nhid))
        self.b1 = Parameter(torch.FloatTensor(1, self.nhid))
        self.l2 = Parameter(torch.FloatTensor(self.nhid, self.nhid))
        self.b2 = Parameter(torch.FloatTensor(1, self.nhid))
        
        self.mlp = Sequential(Linear(self.nhid, self.nhid), Dropout(self.dropout), ReLU(), BatchNorm1d(self.nhid), Linear(self.nhid, self.nhid), Dropout(self.dropout), ReLU(), BatchNorm1d(self.nhid))
        self.NonLinear = NonLinear
        if NonLinear:
            self.p = torch.nn.Parameter(torch.ones(1))
            ## or
            # self.p = torch.nn.Parameter(torch.FloatTensor(1))
            # torch.nn.init.uniform_(self.p.data, -1, 1)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)        
        ones(self.l1)
        zeros(self.b1)
        const(self.l2, 1 / self.nhid)
        zeros(self.b2)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = x.view(-1, self.heads, self.nhid)
        output = self.propagate(edge_index, x=x, num_nodes=x.size(0))
        output = self.mlp(output)
        return output

    def message(self, x_i, x_j, edge_index, num_nodes):
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0], None, num_nodes)
        
        if self.NonLinear:
            p = torch.sigmoid(self.p)*2.
            scale = p*x_j
            row, col = edge_index[0], edge_index[1]
            weight = torch.exp(scale-scale.max())
            deg_div = scatter_add(weight.squeeze()*alpha, row, dim=0, dim_size=num_nodes)
            weights = weight/(deg_div[row,:]+1e-6).unsqueeze(1) 
            x1 = x_j*weights
            output = x1 * alpha.view(-1, self.heads, 1)
        else:
            output = x_j * alpha.view(-1, self.heads, 1)
        
        return output

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads * self.nhid)
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.nhid, self.heads)