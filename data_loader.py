import numpy as np
import scipy.sparse as sp
import scipy.io as scio
import pickle as pkl
import networkx as nx
import sys
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_scipy_sparse_matrix
import numpy as np


YALE = 'Yale'
UMIST = 'UMIST'
THREE_RINGS = 'three_rings'

def load_ogbn_arxiv():
    # Load the OGBN-Arxiv dataset
    dataset_name = 'ogbn-arxiv'
    dataset = PygNodePropPredDataset(name=dataset_name, root='data/')
    data = dataset[0]  # Get the graph data object
    split_idx = dataset.get_idx_split()  # Get train/val/test splits

    # Adjacency matrix
    full_adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)

    # Features (node features)
    features = data.x.numpy()  # Convert PyTorch tensor to numpy array

    # Labels
    labels = data.y.squeeze().numpy()  # Convert to 1D array

    # Train/validation/test indices
    train_index = split_idx['train']
    val_index = split_idx['valid']
    test_index = split_idx['test']

    return full_adj, data.num_nodes, features, labels, train_index, val_index, test_index


def load_cora():
    path = 'data/cora/'
    data_name = 'cora'
    print('Loading from raw data file...')
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, data_name), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    _, _, labels = np.unique(idx_features_labels[:, -1], return_index=True, return_inverse=True)

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, data_name), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj.T + adj
    adj = adj.minimum(1)
    return features, idx_map, adj, labels


def load_citeseer():
    path = 'data/citeseer/'
    data_name = 'citeseer'
    print('Loading from raw data file...')
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, data_name), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    _, _, labels = np.unique(idx_features_labels[:, -1], return_index=True, return_inverse=True)

    idx = np.array(idx_features_labels[:, 0], dtype=np.str)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, data_name), dtype=np.str)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    rows_to_delete = []
    for i in range(edges_unordered.shape[0]):
        if edges[i, 0] is None or edges[i, 1] is None:
            rows_to_delete.append(i)
    edges = np.array(np.delete(edges, rows_to_delete, axis=0), dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj.T + adj
    adj = adj.minimum(1)
    return features, idx_map, adj, labels


def load_pubmed():
    print('Loading from raw data file...')
    data = scio.loadmat('data/pubmed.mat')
    adj = data['W']
    # adj = adj - adj.diagonal()
    features = data['fea']
    # adj = sp.coo_matrix(adj)
    labels = data['gnd']
    labels = np.reshape(labels, (labels.shape[0],))
    adj = adj.T + adj
    adj = adj.minimum(1)
    return features, adj.tocoo(), labels


def load_citeseer_from_mat():
    print('Loading from raw data file...')
    data = scio.loadmat('data/citeseer.mat')
    adj = data['W']
    features = data['fea']
    labels = data['gnd']
    labels = np.reshape(labels, (labels.shape[0],))

    # Convert adjacency to sparse matrix
    adj = sp.coo_matrix(adj)

    # Ensure symmetry and convert to binary (0 or 1)
    adj = adj + adj.T
    adj = adj.minimum(1)

    return features, adj.tocsr(), labels


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/node/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/node/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    features = features.toarray()
    # label = (y_train + y_val + y_test).argmax(axis=1)
    label = labels.argmax(axis=1)
    # return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    return adj, features, label, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict
