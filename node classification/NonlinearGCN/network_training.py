from __future__ import division
from __future__ import print_function

import time
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
import torch.nn.functional as F
import torch.optim as optim
from utils import accuracy


class Training:
    def __init__(self, model, args, device):
        self.device = device
        self.args = args
        self.log_every = 50
        self.best_val_acc = 0
        self.best_val_loss = 100
        self.best_p = None
        self.weights = None
        self.estimator = None
        self.model = model.to(device)

    def fit(self, features, adj, edge, T, labels, idx_train, idx_val, idx_test):
        args = self.args
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        estimator = EstimateP(device=self.device).to(self.device)
        self.estimator = estimator
        self.optimizer_P = optim.SGD(estimator.parameters(), momentum=0.9, lr=args.lr_1)
        
        # Train model
        t_total = time.time()
        for epoch in range(args.epochs):
            self.train_outer(epoch, features, adj, edge, T, labels,
                        idx_train, idx_val)
            for i in range(int(args.inner_steps)):
                self.train_inner(epoch, features, adj, edge, T, estimator.estimated_P,
                        labels, idx_train, idx_val)
        
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        print("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)


    def train_inner(self, epoch, features, adj, edge, T, p, labels, idx_train, idx_val):
        args = self.args
        estimator = self.estimator
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(features, adj, edge, T, p)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        self.optimizer.step()
        
        # Evaluate validation set performance separately,
        self.model.eval()
        output = self.model(features, adj, edge, T, p)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        
        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_p = p.detach()
            self.weights = deepcopy(self.model.state_dict())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_p = p.detach()
            self.weights = deepcopy(self.model.state_dict())
            

    def train_outer(self, epoch, features, adj, edge, T, labels, idx_train, idx_val):
        estimator = self.estimator
        args = self.args
        t = time.time()
        estimator.train()
        self.optimizer_P.zero_grad()
        output = self.model(features, adj, edge, T, estimator.estimated_P)
        loss_gcn = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_gcn.backward()
        clip_grad_value_(estimator.parameters(), 3)
        self.optimizer_P.step()
       
        estimator.estimated_P.data.copy_(estimator.estimated_P.data)        
       
        # Evaluate validation set performance separately,
        self.model.eval()
        output = self.model(features, adj, edge, T, estimator.estimated_P)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        
        if (epoch + 1) % self.log_every == 0:
            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_nogcn: {:.4f}'.format(loss_gcn.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()),
                  'time: {:.4f}s'.format(time.time() - t))

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_p = estimator.estimated_P.detach()
            self.weights = deepcopy(self.model.state_dict())
            
        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_p = estimator.estimated_P.detach()
            self.weights = deepcopy(self.model.state_dict())


    def test(self, features, adj, edge, T, labels, idx_test):
        print("\t=== testing ===")
        self.model.eval()
        p = self.best_p
        print(p)
        if self.best_p is None:
            p = estimator.estimated_P
        output = self.model(features, adj, edge, T, p)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("\tTest set results:", "loss= {:.4f}".format(loss_test.item()), "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()


class EstimateP(nn.Module):

    def __init__(self, device='cpu'):
        super(EstimateP, self).__init__()
        self.estimated_P = nn.Parameter(torch.FloatTensor(1))
        torch.nn.init.constant_(self.estimated_P.data, 1.) 
        #torch.nn.init.uniform_(self.estimated_P.data, -1., 1.)

    def forward(self):
        return self.estimated_P