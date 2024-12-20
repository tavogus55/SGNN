import torch
from model import *
from data_loader import *
import warnings
import scipy.io as scio
import json
from datetime import datetime

warnings.filterwarnings('ignore')
utils.set_seed(0)

print(torch.version.cuda)  # Check the CUDA version supported by PyTorch
print(torch.cuda.is_available())  # Check if CUDA is detected
print(torch.version.__version__)  # Check PyTorch version


decision = input("Choose which dataset to use\n1. Cora\n2. Citeseer\n3. Pubmed\n\nYour input: ")
dataset_name = None

if decision == "1":
    dataset_name = "cora"
    adjacency, features, labels, train_mask, val_mask, test_mask = load_data(dataset_name)
elif decision == "2":
    dataset_name = "citeseer"
    adjacency, features, labels, train_mask, val_mask, test_mask = load_data(dataset_name)
elif decision == "3":
    dataset_name = "pubmed"
    adjacency, features, labels, train_mask, val_mask, test_mask = load_data(dataset_name)
else:
    print("Invalid")
    exit()


# train_mask = np.array([True]*features.shape[0])
# features, adjacency, labels = load_citeseer_from_mat()
# features, adjacency, labels = load_pubmed()
n_class = np.unique(labels).shape[0]
if type(features) is not np.ndarray:
    features = features.todense()
features = torch.Tensor(features)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the JSON settings
with open('config.json', 'r') as file:
    settings = json.load(file)
config = settings["Node Classification"][dataset_name]

# ========== training setting ==========
eta = config["eta"]
BP_count = config["BP_count"]

features = features.to(device)
layer_config = config["layers"]
lam = eval(config["lam"].replace("^", "**"))

# ========== layers setting ==========
relu_func = Func(torch.nn.functional.relu)
linear_func = Func(None)
sigmoid_func = Func(torch.nn.functional.sigmoid)
leaky_relu_func = Func(torch.nn.functional.leaky_relu, negative_slope=0.2)
tanh = Func(torch.nn.functional.tanh)

if dataset_name == "cora":
    layers = [
        LayerParam(layer_config[0]["neurons"], inner_act=linear_func, act=leaky_relu_func, gnn_type=LayerParam.EGCN,
                   learning_rate=eval(layer_config[0]["learning_rate"].replace("^", "**")),
                   order=layer_config[0]["order"], max_iter=layer_config[0]["max_iter"],
                   lam=lam,batch_size=layer_config[0]["batch_size"]),
        LayerParam(layer_config[1]["neurons"], inner_act=linear_func, act=relu_func, gnn_type=LayerParam.EGCN,
                   learning_rate=eval(layer_config[1]["learning_rate"].replace("^", "**")),
                   order=layer_config[1]["order"], max_iter=layer_config[1]["max_iter"], lam=lam,
                   batch_size=layer_config[1]["batch_size"]),
        LayerParam(layer_config[2]["neurons"], inner_act=linear_func, act=linear_func, gnn_type=LayerParam.EGCN,
                   learning_rate=eval(layer_config[2]["learning_rate"].replace("^", "**")),
                   order=layer_config[2]["order"], max_iter=layer_config[2]["max_iter"], lam=lam,
                   batch_size=layer_config[2]["batch_size"]),
    ]

elif dataset_name == "citeseer":
    layers = [
        LayerParam(layer_config[0]["neurons"], inner_act=relu_func, act=leaky_relu_func, gnn_type=LayerParam.EGCN,
                   learning_rate=eval(layer_config[0]["learning_rate"].replace("^", "**")),
                   order=layer_config[0]["order"], max_iter=layer_config[0]["max_iter"],
                   lam=lam, batch_size=layer_config[0]["batch_size"]),
        LayerParam(layer_config[1]["neurons"], inner_act=relu_func, act=linear_func, gnn_type=LayerParam.EGCN,
                   learning_rate=eval(layer_config[1]["learning_rate"].replace("^", "**")),
                   order=layer_config[1]["order"], max_iter=layer_config[1]["max_iter"],
                   lam=lam, batch_size=layer_config[1]["batch_size"]),
    ]

elif dataset_name == "pubmed":
    layers = [
        LayerParam(layer_config[0]["neurons"], inner_act=relu_func, act=leaky_relu_func, gnn_type=LayerParam.EGCN,
                   learning_rate=eval(layer_config[0]["learning_rate"].replace("^", "**")),
                   order=layer_config[0]["order"], max_iter=layer_config[0]["max_iter"],
                   lam=lam, batch_size=layer_config[0]["batch_size"]),
        LayerParam(layer_config[1]["neurons"], inner_act=relu_func, act=leaky_relu_func, gnn_type=LayerParam.EGCN,
                   learning_rate=eval(layer_config[1]["learning_rate"].replace("^", "**")),
                   order=layer_config[1]["order"], max_iter=layer_config[1]["max_iter"],
                   lam=lam, batch_size=layer_config[1]["batch_size"]),
    ]

# ========== overlook setting ==========
overlook_rates = None

sgnn = SupervisedStackedGNN(features, adjacency, layers,
                            training_mask=train_mask, val_mask=test_mask,
                            overlooked_rates=overlook_rates,
                            BP_count=BP_count, eta=eta, device=device,
                            labels=labels, metric_func=utils.classification)

utils.print_SGNN_info(sgnn)

print('============ Start Training ============')
start_time = datetime.now()
prediction = sgnn.run()
print('============ End Training ============')

utils.print_SGNN_info(sgnn)

# ========== Testing ==========
print('============ Start testing ============')
utils.classification(prediction, labels, train_mask)
utils.classification(prediction, labels, val_mask)
utils.classification(prediction, labels, test_mask)
finish_time = datetime.now()

time_difference = finish_time - start_time

# Extract hours, minutes, and seconds
total_seconds = int(time_difference.total_seconds())
hours, remainder = divmod(total_seconds, 3600)
minutes, seconds = divmod(remainder, 60)

print(start_time.strftime("Process started at: " + "%Y-%m-%d %H:%M:%S"))
print(finish_time.strftime("Process started at: " + "%Y-%m-%d %H:%M:%S"))
print(f"Training lasted {hours} hours, {minutes} minutes, {seconds} seconds")
