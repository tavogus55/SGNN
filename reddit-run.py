import torch
from model import *
from data_loader import *
import warnings
from input_data import load_data
from reddit_utils import loadRedditFromNPZ, load_ogbn_arxiv
from datetime import datetime

warnings.filterwarnings('ignore')
# ========== load data ==========
adj, feats, labels, train_index, val_index, test_index = loadRedditFromNPZ('data/')

# Additional variables
num_data = feats.shape[0]
full_adj = adj + adj.T

_ = None
adjacency = full_adj
n_clusters = np.unique(labels).shape[0]
features = torch.Tensor(feats)
feats = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
features = features.to(device)

# ========== training setting ==========

learning_rate = 10**-4
# max_iter = 250000
max_iter = 10000

batch_size = 512


# ========== layers setting ==========
# layers = [128, 64]
relu_func = Func(torch.nn.functional.relu)
linear_func = Func(None)
leaky_relu_func = Func(torch.nn.functional.leaky_relu, negative_slope=0.2)


lam = 10**-6

layers = [
    LayerParam(128, inner_act=linear_func, act=leaky_relu_func, gnn_type=LayerParam.GAE,
               mask_rate=0.2, learning_rate=learning_rate, lam=lam, max_iter=max_iter, batch_size=batch_size),
    LayerParam(64, inner_act=linear_func, act=linear_func, gnn_type=LayerParam.GAE,
               mask_rate=0.2, learning_rate=learning_rate, lam=lam, max_iter=max_iter, batch_size=batch_size),
]


# ========== overlook setting ==========
overlook_rates = None

sgae = StackedGNN(features, adjacency, layers,
                  overlooked_rates=overlook_rates, BP_count=5,
                  eta=10, device=device, labels=labels, metric_func=utils.clustering)

utils.print_SGNN_info(sgae)
start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print('============ Start Training ============')
embedding = sgae.run()
print('============ End Training ============')

utils.print_SGNN_info(sgae)

# ========== Clustering ==========
print('============ Start Clustering ============')
utils.clustering(embedding.cpu().detach().numpy(), labels)
finish_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Start Time: {start_time}")
print(f"Finish Time: {finish_time}")