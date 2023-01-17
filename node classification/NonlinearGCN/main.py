from __future__ import division
from __future__ import print_function
import os
import time
import argparse
import numpy as np
import torch
from model import NonlinearGCN_G, NonlinearGCN_P, NonlinearGCN_S
from network_training import Training
from utils import *


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='pubmed', help='dataset')
parser.add_argument('--aggtype', type=str, default='Softmax', choices=['Generalized-mean', 'Polynomial', 'Softmax'], help='aggtype')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--splits', type=int, default=10, help='Number of data splits.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=400, help='Number of epochs to train.')
parser.add_argument('--inner_steps', type=int, default=2, help='steps for inner optimization')
parser.add_argument('--scale_factor', type=float, default=2., help='scale factor for nonlinear aggregation parameter')
parser.add_argument('--lr_1', type=float, default=0.01, help='lr for training nonlinear aggregation parameter')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")


def main(num):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # load data
    adj, edge, T, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset, num)
    if args.cuda:
        features = features.cuda()
        adj = adj.cuda()
        T = T.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    # load model
    if args.aggtype == 'Generalized-mean':
    	model = NonlinearGCN_G(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item()+1, dropout=args.dropout, s_f=args.scale_factor)
    if args.aggtype == 'Polynomial':
    	model = NonlinearGCN_P(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item()+1, dropout=args.dropout, s_f=args.scale_factor)
    if args.aggtype == 'Softmax':
    	model = NonlinearGCN_S(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item()+1, dropout=args.dropout, s_f=args.scale_factor)

    print("Network Fitting...")
    training = Training(model, args, device)
    training.fit(features, adj, edge, T, labels, idx_train, idx_val, idx_test)
    test_acc = training.test(features, adj, edge, T, labels, idx_test)
    return test_acc


if __name__ == '__main__':
    result = []
    for num in range(args.splits):
        test_acc = main(num)
        result.append(test_acc)
    print('dataset: ', args.dataset, ', ', 'final average result:', np.mean(result), "+", np.std(result))
