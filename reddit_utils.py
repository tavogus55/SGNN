# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collections of preprocessing functions for different graph formats."""

import json
import time

import pandas as pd
from networkx.readwrite import json_graph
import numpy as np
import scipy.sparse as sp
import sklearn.metrics
import sklearn.preprocessing
# import tensorflow.compat.v1 as tf
# from tensorflow.compat.v1 import gfile
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_scipy_sparse_matrix

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

    print(f"Loaded OGBN-Arxiv dataset:")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.edge_index.shape[1]}")
    print(f"Feature matrix shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Train indices: {len(train_index)}")
    print(f"Validation indices: {len(val_index)}")
    print(f"Test indices: {len(test_index)}")

    return full_adj, features, labels, train_index, val_index, test_index



def normalize_features(features):
    """Row-normalize feature matrix."""
    row_sum = np.array(features.sum(1))
    row_inv = np.power(row_sum, -1).flatten()
    row_inv[np.isinf(row_inv)] = 0.0
    row_matrix_inv = sp.diags(row_inv)
    features = row_matrix_inv.dot(features)
    return features


def preprocess_adj(adj):
    """Preprocess adjacency matrix for GCN-based models."""
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(row_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in gfile.Open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def sym_normalize_adj(adj):
    """Normalization by D^{-1/2} (A+I) D^{-1/2}."""
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1)) + 1e-20
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj


def normalize_adj(adj):
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = 1.0 / (np.maximum(1.0, rowsum))
    d_mat_inv = sp.diags(d_inv, 0)
    adj = d_mat_inv.dot(adj)
    return adj


def normalize_adj_diag_enhance(adj, diag_lambda):
    """Normalization by  A'=(D+I)^{-1}(A+I), A'=A'+lambda*diag(A')."""
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = 1.0 / (rowsum + 1e-20)
    d_mat_inv = sp.diags(d_inv, 0)
    adj = d_mat_inv.dot(adj)
    adj = adj + diag_lambda * sp.diags(adj.diagonal(), 0)
    return adj


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


def calc_f1(y_pred, y_true, multilabel):
    if multilabel:
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = 0
    else:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    return sklearn.metrics.f1_score(
        y_true, y_pred, average='micro'), sklearn.metrics.f1_score(
        y_true, y_pred, average='macro')


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def preprocess_multicluster(adj,
                            parts,
                            features,
                            y_train,
                            train_mask,
                            num_clusters,
                            block_size,
                            diag_lambda=-1):
    """Generate the batch for multiple clusters."""

    features_batches = []
    support_batches = []
    y_train_batches = []
    train_mask_batches = []
    total_nnz = 0
    np.random.shuffle(parts)
    for _, st in enumerate(range(0, num_clusters, block_size)):
        pt = parts[st]
        for pt_idx in range(st + 1, min(st + block_size, num_clusters)):
            pt = np.concatenate((pt, parts[pt_idx]), axis=0)
        features_batches.append(features[pt, :])
        y_train_batches.append(y_train[pt, :])
        support_now = adj[pt, :][:, pt]
        if diag_lambda == -1:
            support_batches.append(sparse_to_tuple(normalize_adj(support_now)))
        else:
            support_batches.append(
                sparse_to_tuple(normalize_adj_diag_enhance(support_now, diag_lambda)))
        total_nnz += support_now.count_nonzero()

        train_pt = []
        for newidx, idx in enumerate(pt):
            if train_mask[idx]:
                train_pt.append(newidx)
        train_mask_batches.append(sample_mask(train_pt, len(pt)))
    return (features_batches, support_batches, y_train_batches,
            train_mask_batches)
#
#
# def load_graphsage_data(dataset_str, normalize=True):
#     """Load GraphSAGE data."""
#     start_time = time.time()
#     dataset_path = 'data'
#
#     graph_json = json.load(
#         gfile.Open('{}/{}/{}-G.json'.format(dataset_path, dataset_str,
#                                             dataset_str)))
#
#     print(f"Number of nodes in graph_json: {len(graph_json.get('nodes', []))}")
#     print(f"Number of edges in graph_json: {len(graph_json.get('links', []))}")
#
#     graph_nx = json_graph.node_link_graph(graph_json)
#
#     print(f"Number of nodes in graph_nx: {len(graph_nx.nodes())}")
#     print(f"Number of edges in graph_nx: {len(graph_nx.edges())}")
#
#     id_map = json.load(
#         gfile.Open('{}/{}/{}-id_map.json'.format(dataset_path, dataset_str,
#                                                  dataset_str)))
#     is_digit = list(id_map.keys())[0].isdigit()
#     id_map = {(int(k) if is_digit else k): int(v) for k, v in id_map.items()}
#
#     print(f"Number of nodes in id_map: {len(id_map)}")
#     print(f"First 10 entries in id_map: {list(id_map.items())[:10]}")
#
#     class_map = json.load(
#         gfile.Open('{}/{}/{}-class_map.json'.format(dataset_path, dataset_str,
#                                                     dataset_str)))
#
#     is_instance = isinstance(list(class_map.values())[0], list)
#     class_map = {(int(k) if is_digit else k): (v if is_instance else int(v))
#                  for k, v in class_map.items()}
#
#     broken_count = 0
#     to_remove = []
#     for node in graph_nx.nodes():
#         if node not in id_map:
#             to_remove.append(node)
#             broken_count += 1
#     for node in to_remove:
#         graph_nx.remove_node(node)
#     tf.logging.info(
#         'Removed %d nodes that lacked proper annotations due to networkx versioning issues',
#         broken_count)
#
#     feats = np.load(
#         gfile.Open(
#             '{}/{}/{}-feats.npy'.format(dataset_path, dataset_str, dataset_str),
#             'rb')).astype(np.float32)
#
#     tf.logging.info('Loaded data (%f seconds).. now preprocessing..',
#                     time.time() - start_time)
#     start_time = time.time()
#
#     print(f"Number of nodes in graph_json: {len(graph_json.get('nodes', []))}")
#     print(f"Number of edges in graph_json: {len(graph_json.get('links', []))}")
#     print(f"Number of nodes in graph_nx: {len(graph_nx.nodes())}")
#     print(f"Number of edges in graph_nx: {len(graph_nx.edges())}")
#     print(f"Number of nodes in id_map: {len(id_map)}")
#     print(f"First 10 entries in id_map: {list(id_map.items())[:10]}")
#
#     reversed_id_map = {v: k for k, v in id_map.items()}
#
#     edges = []
#     for edge in graph_nx.edges():
#         if edge[0] in id_map and edge[1] in id_map:
#             edges.append((id_map[edge[0]], id_map[edge[1]]))
#     num_data = len(id_map)
#
#     val_data = np.array(
#         [id_map[n] for n in graph_nx.nodes() if graph_nx.node[n]['val']],
#         dtype=np.int32)
#     test_data = np.array(
#         [id_map[n] for n in graph_nx.nodes() if graph_nx.node[n]['test']],
#         dtype=np.int32)
#     is_train = np.ones((num_data), dtype=np.bool)
#     is_train[val_data] = False
#     is_train[test_data] = False
#     train_data = np.array([n for n in range(num_data) if is_train[n]],
#                           dtype=np.int32)
#
#     train_edges = [
#         (e[0], e[1]) for e in edges if is_train[e[0]] and is_train[e[1]]
#     ]
#     edges = np.array(edges, dtype=np.int32)
#     train_edges = np.array(train_edges, dtype=np.int32)
#
#     # Process labels
#     if isinstance(list(class_map.values())[0], list):
#         num_classes = len(list(class_map.values())[0])
#         labels = np.zeros((num_data, num_classes), dtype=np.float32)
#         for k in class_map.keys():
#             labels[id_map[k], :] = np.array(class_map[k])
#     else:
#         num_classes = len(set(class_map.values()))
#         labels = np.zeros((num_data, num_classes), dtype=np.float32)
#         for k in class_map.keys():
#             labels[id_map[k], class_map[k]] = 1
#
#     if normalize:
#         train_ids = np.array([
#             id_map[n]
#             for n in graph_nx.nodes()
#             if not graph_nx.node[n]['val'] and not graph_nx.node[n]['test']
#         ])
#         train_feats = feats[train_ids]
#         scaler = sklearn.preprocessing.StandardScaler()
#         scaler.fit(train_feats)
#         feats = scaler.transform(feats)
#
#     def _construct_adj(edges, num_data):
#         if len(edges) == 0:
#             print("Warning: No edges found. Returning an empty adjacency matrix.")
#             return sp.csr_matrix((num_data, num_data), dtype=np.float32)
#
#         adj = sp.csr_matrix(
#             (np.ones((edges.shape[0]), dtype=np.float32), (edges[:, 0], edges[:, 1])),
#             shape=(num_data, num_data)
#         )
#         adj += adj.transpose()
#         return adj
#
#     train_adj = _construct_adj(train_edges, num_data)
#     full_adj = _construct_adj(edges, num_data)
#
#     train_feats = feats[train_data]
#     test_feats = feats
#     labels = np.argmax(labels, 1)
#     labels = labels.reshape((labels.shape[0],))
#
#     tf.logging.info('Data loaded, %f seconds.', time.time() - start_time)
#     return num_data, train_adj, full_adj, feats, train_feats, test_feats, labels, train_data, val_data, test_data


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def loadRedditFromNPZ(dataset_dir):
    # Load adjacency matrix
    adj = sp.load_npz(dataset_dir + "reddit_graph.npz")

    # Load data from reddit_data.npz
    data = np.load(dataset_dir + "reddit_data.npz", allow_pickle=True)

    # Extract features and labels
    features = data['feature']  # Assuming 'feature' corresponds to node features
    labels = data['label']  # Assuming 'label' corresponds to node labels

    # Assign dummy train, val, and test indices (adjust based on your needs)
    num_nodes = features.shape[0]
    train_index = np.arange(int(0.6 * num_nodes))  # 60% of nodes for training
    val_index = np.arange(int(0.6 * num_nodes), int(0.8 * num_nodes))  # 20% for validation
    test_index = np.arange(int(0.8 * num_nodes), num_nodes)  # 20% for testing

    return adj, features, labels, train_index, val_index, test_index


def load_reddit_data(normalization="AugNormAdj", cuda=True):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("data/")
    labels = np.zeros(adj.shape[0])
    labels[train_index] = y_train
    labels[val_index] = y_val
    labels[test_index] = y_test
    adj = adj + adj.T
    train_adj = adj[train_index, :][:, train_index]
    features = torch.FloatTensor(np.array(features))
    features = (features - features.mean(dim=0)) / features.std(dim=0)
    return adj, train_adj, features, labels, train_index, val_index, test_index


def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def fetch_normalization(type):
    switcher = {
        'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    }
    func = switcher.get(type, lambda: "Invalid normalization technique.")
    return func
