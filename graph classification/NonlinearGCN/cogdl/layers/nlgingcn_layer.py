import torch
import torch.nn as nn
import torch.nn.functional as F
from cogdl.utils import spmm, get_activation
from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops, degree


class GCNConv_G(nn.Module):
    def __init__(self, apply_func=None, eps=0, train_eps=True, NonLinear=True):
        super(GCNConv_G, self).__init__()
        if train_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([eps]))
        if NonLinear:
            self.p = torch.nn.Parameter(torch.FloatTensor(1))
            torch.nn.init.uniform_(self.p.data, -1, 1)
        self.NonLinear = NonLinear
        self.apply_func = apply_func

    def forward(self, graph, x):
        #edge_index, _ = add_self_loops(torch.cat((graph.edge_index[0].unsqueeze(0),graph.edge_index[1].unsqueeze(0)),dim=0), num_nodes=x.size(0))
        row, col = graph.edge_index
        deg = degree(col, x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        edge = torch.cat((graph.edge_index[0].unsqueeze(0),graph.edge_index[1].unsqueeze(0)),dim=0)
        adj = torch.sparse_coo_tensor(edge,norm,(x.size(0),x.size(0)))
        if self.NonLinear:
            p = torch.sigmoid(self.p)+1.
            mu = x.min()
            x1 = torch.pow(x-mu+1e-6, p)
            out = torch.pow(torch.spmm(adj, x1)+1e-6, 1./p) + (1 + self.eps)*x + mu  
        else:
            out = torch.spmm(adj, x) + (1 + self.eps) * x 
        if self.apply_func is not None:
            out = self.apply_func(out)
        return out


class GCNConv_P(nn.Module):
    def __init__(self, apply_func=None, eps=0, train_eps=True, NonLinear=True):
        super(GCNConv_P, self).__init__()
        if train_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([eps]))
        if NonLinear:
            self.p = torch.nn.Parameter(torch.FloatTensor(1))
            torch.nn.init.uniform_(self.p.data, -1, 1)
        self.NonLinear = NonLinear
        self.apply_func = apply_func

    def forward(self, graph, x):
        #edge_index, _ = add_self_loops(torch.cat((graph.edge_index[0].unsqueeze(0),graph.edge_index[1].unsqueeze(0)),dim=0), num_nodes=x.size(0))
        row, col = graph.edge_index
        deg = degree(col, x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        edge = torch.cat((graph.edge_index[0].unsqueeze(0),graph.edge_index[1].unsqueeze(0)),dim=0)
        adj = torch.sparse_coo_tensor(edge,norm,(x.size(0),x.size(0)))
        if self.NonLinear:
            p = torch.sigmoid(self.p)*2.
            mu = x.min()
            x_top = torch.pow(x-mu+1e-6, p+1)
            x_down = torch.pow(x-mu+1e-6, p)
            out = torch.spmm(adj, x_top)/(torch.spmm(adj, x_down)+1e-6) + (1 + self.eps)*x + mu 
        else:
            out = torch.spmm(adj, x) + (1 + self.eps) * x 
        if self.apply_func is not None:
            out = self.apply_func(out)
        return out


class GCNConv_S(nn.Module):
    def __init__(self, apply_func=None, eps=0, train_eps=True, NonLinear=True):
        super(GCNConv_S, self).__init__()
        if train_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([eps]))
        if NonLinear:
            self.p = torch.nn.Parameter(torch.FloatTensor(1))
            torch.nn.init.uniform_(self.p.data, -1, 1)
        self.NonLinear = NonLinear
        self.apply_func = apply_func

    def forward(self, graph, x):
        #edge_index, _ = add_self_loops(torch.cat((graph.edge_index[0].unsqueeze(0),graph.edge_index[1].unsqueeze(0)),dim=0), num_nodes=x.size(0))
        row, col = graph.edge_index
        deg = degree(col, x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        edge = torch.cat((graph.edge_index[0].unsqueeze(0),graph.edge_index[1].unsqueeze(0)),dim=0)
        adj = torch.sparse_coo_tensor(edge,norm,(x.size(0),x.size(0)))
        if self.NonLinear:
            p = torch.sigmoid(self.p)*2.
            scale = p*x
            softmax = torch.exp(scale-scale.max())
            deg_div = scatter_add(softmax[col,:]*norm.unsqueeze(1), row, dim=0, dim_size=softmax.size()[0])
            weights = softmax[col,:]/(deg_div[row,:]+1e-6) 
            out = scatter_add(weights*x[col,:]*norm.unsqueeze(1), row, dim=0, dim_size=softmax.size()[0]) +(1 + self.eps) * x 
        else:
            out = torch.spmm(adj, x) + (1 + self.eps) * x 
        if self.apply_func is not None:
            out = self.apply_func(out)
        return out