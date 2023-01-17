import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
#from torch_scatter import scatter_max, scatter_add, scatter_mean


class NonlinearGCN_G(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, with_bias=True):
        super(NonlinearGCN_G, self).__init__()
        self.nfeat = nfeat
        self.nclass = nclass
        self.layer1 = GraphConvolution_G(nfeat, nhid, with_bias=with_bias)
        self.layer2 = GraphConvolution(nhid, nclass, with_bias=with_bias)
        self.dropout = dropout

    def forward(self, x, adj, edge, T, p):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.layer1(x, adj, p))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer2(x, adj)
        return F.log_softmax(x, dim=1)


class NonlinearGCN_P(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, with_bias=True):
        super(NonlinearGCN_P, self).__init__()
        self.nfeat = nfeat
        self.nclass = nclass
        self.layer1 = GraphConvolution_P(nfeat, nhid, with_bias=with_bias)
        self.layer2 = GraphConvolution(nhid, nclass, with_bias=with_bias)
        self.dropout = dropout

    def forward(self, x, adj, edge, T, p):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.layer1(x, adj, p))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer2(x, adj)
        return F.log_softmax(x, dim=1)


class NonlinearGCN_S(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, with_bias=True):
        super(NonlinearGCN_S, self).__init__()
        self.nfeat = nfeat
        self.nclass = nclass
        self.layer1 = GraphConvolution_S(nfeat, nhid, with_bias=with_bias)
        self.layer2 = GraphConvolution(nhid, nclass, with_bias=with_bias)
        self.dropout = dropout

    def forward(self, x, adj, edge, T, p):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.layer1(x, T, adj, edge, p))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer2(x, adj)
        return F.log_softmax(x, dim=1)


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolution_G(Module):
    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution_G, self).__init__()
        self.in_features = in_features
        self.out_features = out_features        

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, p):
        p = p+1.  # make p greater than 1
        support = torch.mm(input, self.weight)
        mu = support.min() 
        pre_sup = support - mu  
        pre_in = torch.matmul(adj, torch.pow(pre_sup+1e-6, p))
        pre_out = torch.pow(pre_in+1e-6, 1./p) 
        output = pre_out + mu
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolution_P(Module):
    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution_P, self).__init__()
        self.in_features = in_features
        self.out_features = out_features        
        
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, p): 
        support = torch.mm(input, self.weight)
        mu = torch.min(support)       
        pre_sup = support - mu
        pre_top = torch.pow(pre_sup+1e-6, p+1)
        pre_down = torch.pow(pre_sup+1e-6, p)
        output = torch.matmul(adj, pre_top)/torch.matmul(adj, pre_down) + mu  
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolution_S(Module):
    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution_S, self).__init__()
        self.in_features = in_features
        self.out_features = out_features        
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, T, adj, edge, p):
        p = torch.sigmoid(p)*2.    
        support = torch.mm(input, self.weight)
        e = p * support 
        softmax = torch.exp(e - e.max())
        
        weights = softmax[edge[1],:]/(torch.matmul(adj, softmax)[edge[0],:]+1e-6)
        weights = F.dropout(weights, 0.5, training=self.training)
        output = torch.matmul(T, support[edge[1],:]*weights)

        # val = adj.to_dense()[edge[0],edge[1]]
        # deg=scatter_add(softmax[edge[1],:]*val.unsqueeze(1), edge[0], dim=0, dim_size=softmax.size()[0])
        # weights = (softmax[edge[1],:]*val.unsqueeze(1))/(deg[edge[0],:])
        # weights = F.dropout(weights, 0.6, training=self.training)
        # mask =support[edge[1],:]*weights
        # output = scatter_add(mask, edge[0], dim=0, dim_size=softmax.size()[0])

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'