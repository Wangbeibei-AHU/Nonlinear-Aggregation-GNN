import numpy as np
import scipy.sparse as sp
import torch
import dgl
from dgl import backend as dF
from scipy import io
import pdb


def load_data(dataset, split=0):
    """Load dataset"""
    print('Loading original {} dataset...'.format(dataset))
   
    if dataset == "cora" or dataset == "citeseer" or dataset == "pubmed":
        path = '../data/citation/' + dataset
        adj = io.loadmat(path + '/adj.mat')
        adj = adj['matrix']

        feat = io.loadmat(path + '/feature.mat')
        features = feat['matrix']

        labels = io.loadmat(path + '/label.mat')
        labels = labels['matrix']
        labels = torch.LongTensor(np.where(labels)[1])
        
        idx_train = io.loadmat(path + '/idx_train'+str(split)+'.mat')
        idx_train = idx_train['matrix'].flatten()
        idx_val = io.loadmat(path + '/idx_val'+str(split)+'.mat')
        idx_val = idx_val['matrix'].flatten()
        idx_test = io.loadmat(path + '/idx_test'+str(split)+'.mat')
        idx_test = idx_test['matrix'].flatten()
        #idx_train, idx_val, idx_test = random_planetoid_splits(path, split, labels, labels.max()+1)
    
    elif dataset == "computers" or dataset == "photo" or dataset == "cs":
        path = '../data/amazon/' + dataset
        adj = io.loadmat(path + '/adj.mat')
        adj = adj['adj']

        feat = io.loadmat(path + '/features.mat')
        features = feat['feat']

        labels = io.loadmat(path + '/labels.mat')
        labels = labels['label'].flatten()
        labels = torch.LongTensor(labels)
        idx_train = io.loadmat(path + '/idx_train'+str(split)+'.mat')
        idx_train = idx_train['matrix'].flatten()
        idx_val = io.loadmat(path + '/idx_val'+str(split)+'.mat')
        idx_val = idx_val['matrix'].flatten()
        idx_test = io.loadmat(path + '/idx_test'+str(split)+'.mat')
        idx_test = idx_test['matrix'].flatten()
    
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    edges = adj.nonzero()
    edge = torch.FloatTensor(np.array(adj.todense())).nonzero().t()
    e0 = np.array(edges[0])
    e1 = np.array(edges[1])
    g = dgl.graph((e0, e1))
    g.ndata['feat'] = dF.tensor(features.todense(), dtype=dF.data_type_dict['float32'])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return g, edge, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum + 1e-9, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
    np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def training_performance(val_loss, best_val, acc, best_acc, epoch, best_epoch):
    if val_loss < best_val:
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            best_val = val_loss
    return best_val, best_acc, best_epoch



