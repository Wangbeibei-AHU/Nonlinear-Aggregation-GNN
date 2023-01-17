import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .. import BaseModel
from cogdl.layers import MLP
from cogdl.layers import GINGCNLayer
from cogdl.utils import split_dataset_general
from cogdl.utils import get_activation

class FC(nn.Module):
    def __init__(self, in_feats, out_features, bias=True):
        super(FC, self).__init__()
        self.w = nn.Parameter(torch.zeros(size=(in_feats, out_features)))
        self.bias = nn.Parameter(torch.FloatTensor(1,out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        stdv = 1. / math.sqrt(1)
        self.bias.data.uniform_(-stdv, stdv)


    def forward(self,x):
        if self.bias is not None:
            return torch.matmul(x,self.w)+self.bias
        else:
            return torch.matmul(x,self.w)

# class MLP(nn.Module):
#     def __init__(
#         self,
#         in_feats,
#         out_feats,
#         hidden_size,
#         num_layers,
#         dropout=0.0,
#         activation="relu",
#         norm=None,
#         act_first=False,
#         bias=True
#     ):
#         super(MLP, self).__init__()
#         self.norm = norm
#         self.activation = get_activation(activation)
#         self.act_first = act_first
#         self.dropout = dropout
#         shapes = [in_feats] + [hidden_size] * (num_layers - 1) + [out_feats]
#         self.mlp = nn.ModuleList(
#             [FC(shapes[layer], shapes[layer + 1], bias=bias) for layer in range(num_layers)]
#         )
#         if norm is not None and num_layers > 1:
#             if norm == "layernorm":
#                 self.norm_list = nn.ModuleList(nn.LayerNorm(x) for x in shapes[1:-1])
#             elif norm == "batchnorm":
#                 self.norm_list = nn.ModuleList(nn.BatchNorm1d(x) for x in shapes[1:-1])
#             else:
#                 raise NotImplementedError(f"{norm} is not implemented in CogDL.")
#         self.reset_parameters()

#     def reset_parameters(self):
#         for layer in self.mlp:
#             layer.reset_parameters()
#         if hasattr(self, "norm_list"):
#             for n in self.norm_list:
#                 n.reset_parameters()

#     def forward(self, x):
#         for i, fc in enumerate(self.mlp[:-1]):
#             x = fc(x)
#             if self.act_first:
#                 x = self.activation(x)
#             if self.norm:
#                 s1,s2,s3 = x.shape[0], x.shape[1], x.shape[2]
#                 x = self.norm_list[i](x.reshape(s1*s2,-1))
#                 x = x.reshape(s1,s2,-1)
#             if not self.act_first:
#                 x = self.activation(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.mlp[-1](x)
#         return x

class GINGCN(BaseModel):
    r"""Graph Isomorphism Network from paper `"How Powerful are Graph
    Neural Networks?" <https://arxiv.org/pdf/1810.00826.pdf>`__.

    Args:
        num_layers : int
            Number of GIN layers
        in_feats : int
            Size of each input sample
        out_feats : int
            Size of each output sample
        hidden_dim : int
            Size of each hidden layer dimension
        num_mlp_layers : int
            Number of MLP layers
        eps : float32, optional
            Initial `\epsilon` value, default: ``0``
        pooling : str, optional
            Aggregator type to use, default:　``sum``
        train_eps : bool, optional
            If True, `\epsilon` will be a learnable parameter, default: ``True``
    """

    @staticmethod
    def add_args(parser):
        parser.add_argument("--epsilon", type=float, default=0.0)
        parser.add_argument("--hidden-size", type=int, default=32)
        parser.add_argument("--num-layers", type=int, default=3)
        parser.add_argument("--num-mlp-layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--train-epsilon", dest="train_epsilon", action="store_false")
        parser.add_argument("--pooling", type=str, default="sum")

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_layers,
            args.num_features,
            args.num_classes,
            args.hidden_size,
            args.num_mlp_layers,
            args.epsilon,
            args.pooling,
            args.train_epsilon,
            args.dropout,
        )

    @classmethod
    def split_dataset(cls, dataset, args):
        return split_dataset_general(dataset, args)

    def __init__(
        self,
        num_layers,
        in_feats,
        out_feats,
        hidden_dim,
        num_mlp_layers,
        eps=0,
        pooling="sum",
        train_eps=False,
        dropout=0.5,
    ):
        super(GINGCN, self).__init__()
        self.gin_layers = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.num_layers = num_layers
        hidden_dim=32

        for i in range(num_layers - 1):
            if i == 0:
                mlp = MLP(in_feats, hidden_dim, hidden_dim, num_mlp_layers, norm="batchnorm")
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim, num_mlp_layers, norm="batchnorm")
            self.gin_layers.append(GINGCNLayer(mlp, eps, train_eps))
            self.batch_norm.append(nn.BatchNorm1d(hidden_dim))

        self.linear_prediction = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.linear_prediction.append(nn.Linear(in_feats, out_feats))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, out_feats))
        self.dropout = nn.Dropout(dropout)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, batch):
        h = batch.x
        device = h.device
        batchsize = int(torch.max(batch.batch)) + 1

        layer_rep = [h]
        for i in range(self.num_layers - 1):
            h = self.gin_layers[i](batch, h)
            h = self.batch_norm[i](h)
            h = F.relu(h)
            layer_rep.append(h)

        final_score = 0

        for i in range(self.num_layers):
            hsize = layer_rep[i].shape[1]
            output = torch.zeros(batchsize, layer_rep[i].shape[1]).to(device)
            pooled = output.scatter_add_(dim=0, index=batch.batch.view(-1, 1).repeat(1, hsize), src=layer_rep[i])
            final_score += self.dropout(self.linear_prediction[i](pooled))

        return final_score
