import torch
import torch.nn as nn

from cogdl.utils import spmm, get_activation
from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops, degree


class GINGCNLayer(nn.Module):
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

    def __init__(self, apply_func=None, eps=0, train_eps=True):
        super(GINGCNLayer, self).__init__()
        if train_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([eps]))
        self.apply_func = apply_func

    def forward(self, graph, x):
        #edge_index, _ = add_self_loops(graph.edge_index, num_nodes=x.size(0))
        row, col = graph.edge_index
        deg = degree(col, x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        edge = torch.cat((graph.edge_index[0].unsqueeze(0),graph.edge_index[1].unsqueeze(0)),dim=0)
        adj = torch.sparse_coo_tensor(edge,norm,(x.size(0),x.size(0)))
        out = torch.spmm(adj, x)+(1 + self.eps) * x 
        if self.apply_func is not None:
            out = self.apply_func(out)
        return out
