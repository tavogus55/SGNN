from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_scipy_sparse_matrix
import numpy as np

def load_ogbn_arxiv():
    # Load the OGBN-Arxiv dataset
    dataset_name = 'ogbn-products'
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

    return data.num_nodes, full_adj, features, labels, train_index, val_index, test_index


# Call the function to load the dataset
num_data, full_adj, feats, labels, train_index, val_index, test_index = load_ogbn_arxiv()

print(f"Number of nodes: {num_data}")
print(f"Adjacency matrix shape: {full_adj.shape}")
print(f"Features shape: {feats.shape}")
print(f"Labels shape: {labels.shape}")
