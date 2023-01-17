import numpy as np
import scipy.sparse as sp
import torch
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
        idx_test_full = range(adj.shape[0])
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
        idx_test_full = range(adj.shape[0])
        #idx_train, idx_val, idx_test = random_coauthor_amazon_splits(path, split, labels, labels.max()+1, seed)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) 
    adj = normalize_adj(adj +sp.eye(adj.shape[0]))
    edge = adj.nonzero()
    T = create_transition_matrix(np.array(adj.todense()))
    T = sparse_mx_to_torch_sparse_tensor(T)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    features = normalize_features(features)
    features = torch.FloatTensor(np.array(features.todense()))
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj, edge, T, features, labels, idx_train, idx_val, idx_test


def random_planetoid_splits(path, split, labels, num_classes):
    indices = []
    for i in range(num_classes):
        index = torch.nonzero(labels == i).view(-1)
        #index = torch.nonzero(labels == i, as_tuple=False).view(-1)
        index = index[torch.randperm(index.size(0))]#, generator=g
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]#
    
    idx_train = train_index
    idx_val = rest_index[:500]
    idx_test = rest_index[500:1500]

    return idx_train, idx_val, idx_test


def random_coauthor_amazon_splits(path, split, labels, num_classes, seed):
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

    indices = []
    for i in range(num_classes):
        index = torch.nonzero(labels == i).view(-1)#, as_tuple=False
        index = index[torch.randperm(index.size(0), generator=g)]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)
    rest_index = torch.cat([i[50:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0), generator=g)]

    idx_train = train_index
    idx_val = val_index
    idx_test = rest_index
   
    return idx_train, idx_val, idx_test

        
def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)) + 1e-10
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)) + 1e-10
    r_inv = np.power(rowsum, -1).flatten()
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


def create_transition_matrix(mx):
    '''create n * e transition matrix T'''
    edge_index = np.nonzero(mx)
    num_edge = int(len(edge_index[0]))
    
    row_index = edge_index[0]
    col_index = range(num_edge)
    data = mx.ravel()[np.flatnonzero(mx)]
    T = sp.csr_matrix((data, (row_index, col_index)),
               shape=(mx.shape[0], num_edge))
    return T    
