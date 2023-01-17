import torch
import torch.nn as nn
import torch.nn.functional as F
from cogdl.utils import EdgeSoftmax
from cogdl.utils import spmm
from torch_scatter import scatter_add


class GATLayer(nn.Module):
    def __init__(self, mlp, in_features, head=4, eps=0,train_eps=True,dropout=0.0, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.alpha = alpha
        self.head = head

        self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=self.head, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=in_features, out_channels=self.head, kernel_size=1, stride=1, padding=0)
        
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.edge_softmax = EdgeSoftmax() 

        self.mlp = mlp

        if train_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([eps]))

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


    def forward(self, graph, input): 
        edge = torch.cat((graph.edge_index[0].unsqueeze(0),graph.edge_index[1].unsqueeze(0)),dim=0)
        #-----new--------------
        N = input.size()[0]
        x = self.mlp(input)
        #-------same a----------------
        h = input.t().unsqueeze(0)
        h1 = self.conv1(h).squeeze(0)
        h2 = self.conv2(h).squeeze(0)

        value = h1[:,edge[0,:]]+h2[:,edge[1,:]]
        # edge_e = torch.exp(-self.leakyrelu(value))
        
        # deg = scatter_add(edge_e, edge[0], dim=1, dim_size=N)
        # deg_inv_sqrt = deg.pow(-1)
        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # v = deg_inv_sqrt[:,edge[0]] * edge_e
        
        # v = self.leakyrelu(value)
        # v = self.edge_softmax(graph, v.t())
        # v = self.dropout(v.t())

        v = F.relu(value)#self.leakyrelu(value)
        v = self.dropout(v)

        x1 = x[:,edge[1],:].permute(2,0,1)
        h_prime = scatter_add(x1*v, edge[0], dim=2, dim_size=N).permute(1,2,0) + self.eps*x
        h_prime = h_prime.permute(1,0,2).reshape(N,-1)
        return h_prime


class GINGATLayer(nn.Module):
    r"""Graph Isomorphism Network layer from paper `"How Powerful are Graph
    Neural Networks?" <https://arxiv.org/pdf/1810.00826.pdf>`__.

    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{sum}\left(\left\{h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)

    Parameters
    ----------
    apply_func : callable layer function)
        layer or function applied to update node feature
    eps : float32, optional
        Initial `\epsilon` value.
    train_eps : bool, optional
        If True, `\epsilon` will be a learnable parameter.
    """

    def __init__(self, apply_func=None, eps=0, train_eps=True,infeat=0.,head=4):
        super(GINGATLayer, self).__init__()
        self.apply_func = apply_func
        self.gat = GATLayer(apply_func,infeat,head,eps,train_eps)

    def forward(self, graph, x):
        # out = (1 + self.eps) * x + spmm(graph, x)
        out = self.gat(graph, x)
        return out
