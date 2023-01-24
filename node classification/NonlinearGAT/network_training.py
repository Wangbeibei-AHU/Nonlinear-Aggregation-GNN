import dgl
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils import *
from model import NonlinearGAT_G, NonlinearGAT_P, NonlinearGAT_S, EstimateP


class Training:
    def __init__(self, num, args):
        self.fastmode = False
        self.log_every = 50
        self.seed = args.seed
        self.epochs = args.epochs
        self.lr = args.lr
        self.lr_1 = args.lr_1
        self.weight_decay = args.weight_decay
        self.hidden = args.hidden
        self.nb_heads = args.head
        self.dropout = args.dropout
        self.alpha = args.alpha
        self.s_f = args.scale_factor
        self.dataset = args.dataset
        self.aggtype = args.aggtype
        self.inner = args.inner_steps
        self.num = num
        self.result = 0
        if self.data == 'citeseer':
            if self.aggtype == 'Generalized-mean' or self.aggtype ==  'Polynomial':
                self.lr = 0.02

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

        # load data
        graph, edge, labels, idx_train, idx_val, idx_test = load_data(self.dataset, self.num)
        self.features = graph.ndata['feat']
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.edge = edge

        # load model
        if args.aggtype == 'Generalized-mean':
            self.model = NonlinearGAT_G(nfeat=self.features.shape[1], nhid=self.hidden, nclass=int(self.labels.max()) + 1, dropout=self.dropout, alpha=self.alpha, nheads=self.nb_heads)
        if args.aggtype == 'Polynomial':
            self.model = NonlinearGAT_P(nfeat=self.features.shape[1], nhid=self.hidden, nclass=int(self.labels.max()) + 1, dropout=self.dropout, alpha=self.alpha, nheads=self.nb_heads, s_f=self.s_f)
        if args.aggtype == 'Softmax':
            self.model = NonlinearGAT_S(nfeat=self.features.shape[1], nhid=self.hidden, nclass=int(self.labels.max()) + 1, dropout=self.dropout, alpha=self.alpha, nheads=self.nb_heads, s_f=self.s_f)
        self.estimator = EstimateP(self.nb_heads)

        if torch.cuda.is_available():
            device = torch.device("cuda:%d" % 0)
            self.model.cuda()
            self.estimator.cuda()
            self.features = self.features.cuda()
            self.edge = self.edge.cuda()
            self.labels = self.labels.cuda()
            self.idx_train = self.idx_train.cuda()
            self.idx_val = self.idx_val.cuda()
            self.idx_test = self.idx_test.cuda()
            self.graph = graph.to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer_p = optim.SGD(self.estimator.parameters(), momentum=0.9, lr=self.lr_1)

        t_total = time.time()
        loss_values = []
        best = 0
        best_val = 1e9
        best_epoch = 0

        print("Network Fitting...")
        for epoch in range(self.epochs):
            loss_val, acc_t = self.train_outer(epoch)
            loss_values.append(loss_val)
            best_val, best, best_epoch = training_performance(loss_values[-1], best_val, acc_t, best, epoch, best_epoch)
            for i in range(self.inner):
                loss_val, acc_t = self.train_inner(epoch, self.estimator.p)
                loss_values.append(loss_val)
                best_val, best, best_epoch = training_performance(loss_values[-1], best_val, acc_t, best, epoch, best_epoch)

        total_time = time.time() - t_total
        self.result = best.data
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(total_time))
        print('Semi-supervised node classification accuracy: ' + str(best.data) + ' at epoch '
              + str(best_epoch + 1) + '\n')

    def train_inner(self, epoch, p):
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        predict = self.model(self.graph, self.features, self.edge, p)
        loss_train = F.nll_loss(predict[self.idx_train], self.labels[self.idx_train])
        acc_train = accuracy(predict[self.idx_train], self.labels[self.idx_train])
        loss_train.backward()
        self.optimizer.step()
        torch.cuda.empty_cache()

        if not self.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self.model.eval()
            predict = self.model(self.graph, self.features, self.edge, p)

        loss_val = F.nll_loss(predict[self.idx_val], self.labels[self.idx_val])
        acc_val = accuracy(predict[self.idx_val], self.labels[self.idx_val])
        acc_test = accuracy(predict[self.idx_test], self.labels[self.idx_test])

        return loss_val.data.item(), acc_test

    def train_outer(self, epoch):
        t = time.time()
        estimator = self.estimator
        estimator.train()
        self.optimizer_p.zero_grad()

        predict = self.model(self.graph, self.features, self.edge, estimator.p)
        loss_train = F.nll_loss(predict[self.idx_train], self.labels[self.idx_train])
        acc_train = accuracy(predict[self.idx_train], self.labels[self.idx_train])
        loss_train.backward()
        self.optimizer_p.step()
        torch.cuda.empty_cache()
        estimator.p.data.copy_(estimator.p.data)

        if not self.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self.model.eval()
            predict = self.model(self.graph, self.features, self.edge, estimator.p)

        loss_val = F.nll_loss(predict[self.idx_val], self.labels[self.idx_val])
        acc_val = accuracy(predict[self.idx_val], self.labels[self.idx_val])
        acc_test = accuracy(predict[self.idx_test], self.labels[self.idx_test])

        if (epoch + 1) % self.log_every == 0:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.data.item()),
                  'acc_train: {:.4f}'.format(acc_train.data.item()),
                  'loss_val: {:.4f}'.format(loss_val.data.item()),
                  'acc_val: {:.4f}'.format(acc_val.data.item()),
                  "accuracy= {:.4f}".format(acc_test.data),
                  'time: {:.4f}s'.format(time.time() - t))

        return loss_val.data.item(), acc_test
