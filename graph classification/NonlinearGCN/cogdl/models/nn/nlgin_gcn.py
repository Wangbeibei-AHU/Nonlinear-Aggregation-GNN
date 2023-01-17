import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .. import BaseModel
from cogdl.layers import MLP
from cogdl.layers import GCNConv_G, GCNConv_P, GCNConv_S
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


class NonlinearGCN(BaseModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--epsilon", type=float, default=0.0)
        parser.add_argument("--hidden-size", type=int, default=32)
        parser.add_argument("--num-layers", type=int, default=3)
        parser.add_argument("--num-mlp-layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--train-epsilon", dest="train_epsilon", action="store_false")
        parser.add_argument("--pooling", type=str, default="sum")
        parser.add_argument("--mod", type=str, default="Generalized-mean", choices=['Generalized-mean', 'Polynomial', 'Softmax'])

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
            args.mod,
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
        mod = 'Softmax'
    ):
        super(NonlinearGCN, self).__init__()
        self.gin_layers = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.num_layers = num_layers
        print('num layers: ', self.num_layers)
        print(hidden_dim)
        NonLinear = [True, False, False]
        print('mod: ', mod)

        for i in range(num_layers - 1):
            if i == 0:
                mlp = MLP(in_feats, hidden_dim, hidden_dim, num_mlp_layers, norm="batchnorm")
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim, num_mlp_layers, norm="batchnorm")
            if mod == 'Generalized-mean':
                self.gin_layers.append(GCNConv_G(mlp, eps, train_eps, NonLinear[i]))
            if mod == 'Polynomial':
                self.gin_layers.append(GCNConv_P(mlp, eps, train_eps, NonLinear[i]))
            if mod == 'Softmax':
                self.gin_layers.append(GCNConv_S(mlp, eps, train_eps, NonLinear[i]))
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
