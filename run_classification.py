import torch
from keras_applications.mobilenet_v2 import layers
from tensorboard.compat.tensorflow_stub.dtypes import double

from model import *
from data_loader import *
import warnings
import scipy.io as scio
import json
from datetime import datetime
from reddit_utils import load_reddit_data

warnings.filterwarnings('ignore')
utils.set_seed(0)

print(torch.version.cuda)  # Check the CUDA version supported by PyTorch
print(torch.cuda.is_available())  # Check if CUDA is detected
print(torch.version.__version__)  # Check PyTorch version


def run_classificaton(cuda_num, dataset_choice, config):

    start_time = datetime.now()

    if dataset_choice == "Cora" or dataset_choice == "Citeseer" or dataset_choice == "PubMed":
        adjacency, features, labels, train_mask, val_mask, test_mask = load_data(dataset_choice)
    elif dataset_choice == "Flickr":
        adjacency, features, labels, train_mask, val_mask, test_mask = load_flickr_data(dataset_choice)
    elif dataset_choice == "FacebookPagePage":
        adjacency, features, labels, train_mask, val_mask, test_mask = load_facebook_pagepage_dataset(dataset_choice)
    elif dataset_choice == "Actor":
        adjacency, features, labels, train_mask, val_mask, test_mask = load_actor_dataset(dataset_choice)
    elif dataset_choice == "LastFMAsia":
        adjacency, features, labels, train_mask, val_mask, test_mask = load_lastfmasia_dataset(dataset_choice)
    elif dataset_choice == "DeezerEurope":
        adjacency, features, labels, train_mask, val_mask, test_mask = load_deezereurope_dataset(dataset_choice)
    elif dataset_choice == "Amazon Computers":
        adjacency, features, labels, train_mask, val_mask, test_mask = load_amazon_dataset(dataset_choice.split(" ")[0],
                                                                                           dataset_choice.split(" ")[1])
    elif dataset_choice == "Amazon Photo":
        adjacency, features, labels, train_mask, val_mask, test_mask = load_amazon_dataset(dataset_choice.split(" ")[0],
                                                                                           dataset_choice.split(" ")[1])
    elif dataset_choice == "Reddit":
        adjacency, _, features, labels, train_mask, val_mask, test_mask = load_reddit_data()
    elif dataset_choice == "Arxiv":
        adjacency, _, features, labels, train_mask, val_mask, test_mask = load_ogbn_dataset(dataset_choice.lower())
    elif dataset_choice == "Products":
        adjacency, _, features, labels, train_mask, val_mask, test_mask = load_ogbn_dataset(dataset_choice.lower())
    else:
        print("Invalid dataset")
        exit()


    n_class = np.unique(labels).shape[0]
    if type(features) is not np.ndarray and dataset_choice != "Reddit":
        features = features.todense()
    features = torch.Tensor(features)
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")

    # ========== training setting ==========
    eta = config["eta"]
    BP_count = config["BP_count"]

    features = features.to(device)
    layer_config = config["layers"]
    lam = eval(config["lam"].replace("^", "**"))

    # ========== layers setting ==========


    layer_number = 0
    layers = []


    for layer in layer_config:

        current_layer_activation = layer["activation"]
        current_layer_inner_act = layer["inner_act"]

        chosen_act = get_activation(current_layer_activation)
        chosen_inner_act = get_activation(current_layer_inner_act)

        if dataset_choice == "Reddit" or dataset_choice == "Arxiv" or dataset_choice == "Products":
            layer_to_add = LayerParam(layer["neurons"], inner_act=chosen_inner_act, act=chosen_act,
                                      gnn_type=LayerParam.EGCN,
                                      learning_rate=eval(layer["learning_rate"].replace("^", "**")),
                                      max_iter=layer["max_iter"], lam=lam, batch_size=layer["batch_size"])
        else:
            layer_to_add = LayerParam(layer["neurons"], inner_act=chosen_inner_act, act=chosen_act,
                                      gnn_type=LayerParam.EGCN,
                                      learning_rate=eval(layer["learning_rate"].replace("^", "**")),
                                      order=layer["order"], max_iter=layer["max_iter"],
                                      lam=lam,batch_size=layer["batch_size"])

        layers.append(layer_to_add)
        layer_number = layer_number + 1

    # ========== overlook setting ==========
    overlook_rates = None

    sgnn = SupervisedStackedGNN(features, adjacency, layers,
                                training_mask=train_mask, val_mask=test_mask,
                                overlooked_rates=overlook_rates,
                                BP_count=BP_count, eta=eta, device=device,
                                labels=labels, metric_func=utils.classification)

    utils.print_SGNN_info(sgnn)

    print('============ Start Training ============')
    prediction = sgnn.run()
    print('============ End Training ============')

    utils.print_SGNN_info(sgnn)

    # ========== Testing ==========
    print('============ Start testing ============')
    print("Training accuracy")
    utils.classification(prediction, labels, train_mask)
    print("Validation accuracy")
    utils.classification(prediction, labels, val_mask)
    print("Test accuracy")
    accuracy = utils.classification(prediction, labels, test_mask)
    finish_time = datetime.now()

    time_difference = finish_time - start_time

    # Extract hours, minutes, and seconds
    total_seconds = int(time_difference.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(start_time.strftime("Process started at: " + "%Y-%m-%d %H:%M:%S"))
    print(finish_time.strftime("Process started at: " + "%Y-%m-%d %H:%M:%S"))
    print(f"Training lasted {hours} hours, {minutes} minutes, {seconds} seconds")
    total_max_iter = 0
    for layer in layer_config:
        total_max_iter = total_max_iter + layer["max_iter"]

    total_iterations = total_max_iter*((BP_count*2)+1)
    print(f"Total iterations: {total_iterations}")
    efficiency = total_seconds / total_iterations
    print(f"Official efficiency: {efficiency}")

    return accuracy, efficiency, dataset_choice

def get_activation(current_layer_activation):

    if "tanh" in current_layer_activation:
        chosen_activation = Func(torch.nn.functional.tanh)
    elif "sigmoid" in current_layer_activation:
        chosen_activation = Func(torch.nn.functional.sigmoid)
    elif "linear" in current_layer_activation:
        chosen_activation = Func(None)
    elif "leaky" in current_layer_activation:
        negative_slope = float(current_layer_activation.split("=")[1])
        chosen_activation = Func(torch.nn.functional.leaky_relu, negative_slope=negative_slope)
    elif current_layer_activation == "relu":
        chosen_activation = Func(torch.nn.functional.relu)
    else:
        print("Not activation type set")
        exit()

    return chosen_activation