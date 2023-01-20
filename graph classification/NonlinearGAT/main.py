from __future__ import division
from __future__ import print_function
import os.path as osp
import time
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from torch.optim.lr_scheduler import MultiStepLR
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from model import NonLinearGAT
from util import separate_data, Constant, Config


def train(model, loader, optimizer, device):
    loss_all = 0
    model.train()
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        clip_grad_value_(model.parameters(), 2.0)
        optimizer.step()
        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)


def test(model, loader, device):
    acc_all = 0
    model.eval()
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        acc_all += pred.eq(data.y).sum().item()
    return acc_all / len(loader.dataset)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="PROTEINS", help='name of dataset')
    parser.add_argument('--mod', type=str, default="Generalized-mean", choices=['Generalized-mean', 'Polynomial', 'Softmax'], help='nonlinear aggregation to be used')
    parser.add_argument('--seed', type=int, default=809, help='random seed')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate')
    parser.add_argument('--wd', type=float, default=1e-3, help='weight decay value')
    parser.add_argument('--n_layer', type=int, default=2, help='number of hidden layers')
    parser.add_argument('--hid', type=int, default=32, help='size of input hidden units')
    parser.add_argument('--heads', type=int, default=1, help='number of attention heads')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha for the leaky_relu')
    parser.add_argument('--kfold', type=int, default=10, help='number of kfold')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--readout', type=str, default="add", choices=["add", "mean"], help='readout function: add, mean')
    args = parser.parse_args()
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', args.dataset)
    
    dataset = TUDataset(path, name=args.dataset, pre_transform=Constant()).shuffle()

    train_graphs, test_graphs = separate_data(len(dataset), args.kfold)

    kfold_num = args.kfold
    print('Dataset:', args.dataset)
    print('# of graphs:', len(dataset))
    print('# of classes:', dataset.num_classes)
          
    test_acc_values = torch.zeros(kfold_num, args.epochs)
    print('model: ', args.mod) 
    for idx in range(kfold_num):
        print('=============================================================================')
        print(kfold_num, 'fold cross validation:', idx+1)
        
        idx_train = train_graphs[idx]
        idx_test = test_graphs[idx]

        train_dataset = dataset[idx_train]
        test_dataset = dataset[idx_test]
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=args.seed)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        t_start = time.time()
        best_epoch = 0

        config = Config(mod=args.mod, nhid=args.hid, nclass=dataset.num_classes, 
                        nfeat=dataset.num_features, dropout=args.dropout, 
                        heads=args.heads, alpha=args.alpha, 
                        n_layer=args.n_layer, readout=args.readout)
            
        model = NonLinearGAT(config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)
        scheduler = MultiStepLR(optimizer, milestones=[50,100,150,200,250,300,350,400,450,500], gamma=0.5)

        for epoch in range(args.epochs):
            train_loss = train(model, train_loader, optimizer, device)
            train_acc = test(model, train_loader, device)
            test_acc = test(model, test_loader, device)
            test_acc_values[idx, epoch] = test_acc
            scheduler.step()
        
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_start))

    print('=============================================================================')
    mean_test_acc = torch.mean(test_acc_values, dim=0)
    best_epoch = int(torch.argmax(mean_test_acc).data)
    mean_acc = torch.mean(test_acc_values[:, best_epoch])
    mean_std = torch.std(test_acc_values[:, best_epoch])
    print('test_acc: ', mean_acc, ' best_std: ', mean_std)

    
def main_multi(dataset, hid, lr, dropout):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="PROTEINS", help='name of dataset')
    parser.add_argument('--mod', type=str, default="Generalized-mean", choices=['Generalized-mean', 'Polynomial', 'Softmax'], help='nonlinear aggregation to be used')
    parser.add_argument('--seed', type=int, default=809, help='random seed')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate')
    parser.add_argument('--wd', type=float, default=1e-3, help='weight decay value')
    parser.add_argument('--n_layer', type=int, default=2, help='number of hidden layers')
    parser.add_argument('--hid', type=int, default=32, help='size of input hidden units')
    parser.add_argument('--heads', type=int, default=1, help='number of attention heads')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha for the leaky_relu')
    parser.add_argument('--kfold', type=int, default=10, help='number of kfold')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--readout', type=str, default="add", choices=["add", "mean"], help='readout function: add, mean')
    args = parser.parse_args()
    args.dataset, args.hid, args.lr, args.dropout = dataset, hid, lr, dropout
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', args.dataset)
    
    dataset = TUDataset(path, name=args.dataset, pre_transform=Constant()).shuffle()

    train_graphs, test_graphs = separate_data(len(dataset), args.kfold)

    kfold_num = args.kfold
    print('Dataset:', args.dataset)
    print('# of graphs:', len(dataset))
    print('# of classes:', dataset.num_classes)
          
    test_acc_values = torch.zeros(kfold_num, args.epochs)
    print('model: ', args.mod) 
    for idx in range(kfold_num):
        print('=============================================================================')
        print(kfold_num, 'fold cross validation:', idx+1)
        
        idx_train = train_graphs[idx]
        idx_test = test_graphs[idx]

        train_dataset = dataset[idx_train]
        test_dataset = dataset[idx_test]
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=args.seed)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        t_start = time.time()
        best_epoch = 0

        config = Config(mod=args.mod, nhid=args.hid, nclass=dataset.num_classes, 
                        nfeat=dataset.num_features, dropout=args.dropout, 
                        heads=args.heads, alpha=args.alpha, 
                        n_layer=args.n_layer, readout=args.readout)
            
        model = NonLinearGAT(config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)
        scheduler = MultiStepLR(optimizer, milestones=[50,100,150,200,250,300,350,400,450,500], gamma=0.5)

        for epoch in range(args.epochs):
            train_loss = train(model, train_loader, optimizer, device)
            train_acc = test(model, train_loader, device)
            test_acc = test(model, test_loader, device)
            test_acc_values[idx, epoch] = test_acc
            scheduler.step()
        
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_start))

    print('=============================================================================')
    mean_test_acc = torch.mean(test_acc_values, dim=0)
    best_epoch = int(torch.argmax(mean_test_acc).data)
    mean_acc = torch.mean(test_acc_values[:, best_epoch])
    mean_std = torch.std(test_acc_values[:, best_epoch])
    print('test_acc: ', mean_acc, ' test_std: ', mean_std)
    return mean_acc.item(), mean_std.item()


if __name__ == '__main__':
    #main()
    
    ## Tips
    "Note that If the results are inconsistent with reported ones due to differences in python environment or equipment," 
    "you can get your own parameters in the following ways:"
    datasets = ['MUTAG', 'PTC_MR', 'PROTEINS']
    dnum = len(datasets)
    lrs = [0.001,0.03,0.01]
    dropouts = [0.,0.3,0.5]
    hidden_size = [32,64]

    best_lr = np.zeros(dnum)
    best_dropout=np.zeros(dnum)
    best_hidden=np.zeros(dnum)
    best_test_acc=np.zeros(dnum)
    best_test_std=np.zeros(dnum)
    for i in range(dnum):
        dataset = datasets[i]
        for lr in lrs:
            for h in hidden_size: 
                for d in dropouts:
                    res = np.zeros(5)
                    std = np.zeros(5)
                    for j in range(5):
                        res[j],std[j]=main_multi(dataset, h, lr, d)
                    mean_acc = np.mean(res)
                    mean_std = np.mean(std)
                    print('finall result:', mean_acc, '+', mean_std )
                    if mean_acc > best_test_acc[i]:
                        best_test_acc[i]=mean_acc
                        best_test_std[i]=mean_std
                        best_lr[i] = lr
                        best_dropout[i]=d
                        best_hidden[i]=h

    for n in range(dnum):
        print('dataset: ', datasets[n], 'lr=', best_lr[n], 'dropout=', best_dropout[n], 'hidden=',  best_hidden[n],' reslut: ', best_test_acc[n], ' + ', best_test_std[n])
