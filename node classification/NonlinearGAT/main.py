from __future__ import division
from __future__ import print_function
import os
import argparse
import numpy as np
from network_training import Training

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='citeseer', help='dataset')
parser.add_argument('--aggtype', type=str, default='Polynomial', choices=['Generalized-mean', 'Polynomial', 'Softmax'], help='aggtype')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--splits', type=int, default=10, help='Number of data splits.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--head', type=int, default=8, help='Number of head.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='alpha for the leaky_relu')
parser.add_argument('--epochs', type=int,  default=500, help='Number of epochs to train.')
parser.add_argument('--inner_steps', type=int, default=3, help='steps for inner optimization')
parser.add_argument('--scale_factor', type=float, default=2., help='scale factor for nonlinear aggregation parameter')
parser.add_argument('--lr_1', type=float, default=0.01, help='lr for training nonlinear aggregation parameter')
args = parser.parse_args()


if __name__ == '__main__':
    result = np.zeros(args.splits)
    for num in range(args.splits):
        train = Training(num, args)
        result[num] = train.result
    print('dataset: ', args.dataset, ', ', 'final average result:', np.mean(result), "+", np.std(result))

