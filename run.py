import torch
from model import *
from data_loader import *
import warnings
import scipy.io as scio
import utils
import json
from datetime import datetime

warnings.filterwarnings('ignore')
utils.set_seed(0)

print(torch.version.cuda)  # Check the CUDA version supported by PyTorch
print(torch.cuda.is_available())  # Check if CUDA is detected
print(torch.version.__version__)  # Check PyTorch version


def run_clustering(dataset_choice):
    dataset_name = None
    start_time = datetime.now()
    if dataset_choice == "1":
        dataset_name = "Cora"
        features, _, adjacency, labels = load_cora()
    elif dataset_choice == "2":
        dataset_name = "Citeseer"
        features, adjacency, labels = load_citeseer_from_mat()
    elif dataset_choice == "3":
        dataset_name = "PubMed"
        features, adjacency, labels = load_pubmed()
    elif dataset_choice == "4":
        dataset_name = "Flickr"
        adjacency, features, labels, _, _, _ = load_flickr_data(dataset_name)
    elif dataset_choice == "5":
        dataset_name = "FacebookPagePage"
        adjacency, features, labels, _, _, _ = load_facebook_pagepage_dataset(dataset_name)
    elif dataset_choice == "6":
        dataset_name = "Actor"
        adjacency, features, labels, _, _, _ = load_actor_dataset(dataset_name)
    elif dataset_choice == "7":
        dataset_name = "LastFMAsia"
        adjacency, features, labels, _, _, _ = load_lastfmasia_dataset(dataset_name)
    elif dataset_choice == "8":
        dataset_name = "DeezerEurope"
        adjacency, features, labels, _, _, _ = load_deezereurope_dataset(dataset_name)
    elif dataset_choice == "9":
        dataset_name = "Amazon"
        dataset_type = "Computers"
        adjacency, features, labels, _, _, _ = load_amazon_dataset(dataset_name, dataset_type)
    elif dataset_choice == "10":
        dataset_name = "Amazon"
        dataset_type = "Photo"
        adjacency, features, labels, _, _, _ = load_amazon_dataset(dataset_name, dataset_type)
    else:
        print("Invalid")
        exit()

    # Load the JSON settings
    with open('config.json', 'r') as file:
        settings = json.load(file)
    config = settings["Node Clustering"][dataset_name]

    mask_rate = config["mask_rate"]
    overlook_rates = config["overlook_rates"]
    layers = config["layers"]
    max_iter = config["max_iter"]
    batch_size = config["batch"]
    BP_count = config["BP_count"]
    learning_rate = eval(config["learning_rate"].replace("^", "**"))
    lam = eval(config["lam"].replace("^", "**"))
    eta = config["eta"]
    loss = config["loss"]
    negative_slope = config["negative_slope"]

    # ========== load data ==========
    # features, _, adjacency, labels = load_cora()
    # features, _, adjacency, labels = load_citeseer()
    # features, adjacency, labels = load_citeseer_from_mat()
    # features, adjacency, labels = load_pubmed()
    n_clusters = np.unique(labels).shape[0]
    if type(features) is not np.ndarray:
        features = features.todense()
    features = torch.Tensor(features)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ========== training setting ==========
    features = features.to(device)

    # ========== layers setting ==========
    # layers = [32, 16]
    # layers = [128, 64, 32]  # Cora
    # layers = [256, 128]

    relu_func = Func(torch.nn.functional.relu)
    linear_func = Func(None)
    leaky_relu_func = Func(torch.nn.functional.leaky_relu, negative_slope=negative_slope)

    if (dataset_name == "PubMed" or dataset_name == "Citeseer" or dataset_name == "Flickr"
            or dataset_name == "FacebookPagePage" or dataset_name == "Actor" or dataset_name == "LastFMAsia"
            or dataset_name == "DeezerEurope" or dataset_name == "Amazon"):
        layers = [
            LayerParam(layers[0], inner_act=linear_func, act=leaky_relu_func, gnn_type=LayerParam.GAE,
                       mask_rate=mask_rate, lam=lam, max_iter=max_iter, learning_rate=learning_rate,
                       batch_size=batch_size),
            LayerParam(layers[1], inner_act=linear_func, act=linear_func, gnn_type=LayerParam.GAE,
                       mask_rate=mask_rate, lam=lam, max_iter=max_iter, learning_rate=learning_rate,
                       batch_size=batch_size),
        ]
    else:
        layers = [
            LayerParam(layers[0], inner_act=linear_func, act=leaky_relu_func, gnn_type=LayerParam.GAE,
                       mask_rate=mask_rate, lam=lam, max_iter=max_iter, learning_rate=learning_rate,
                       batch_size=batch_size),
            LayerParam(layers[1], inner_act=linear_func, act=leaky_relu_func, gnn_type=LayerParam.GAE,
                       mask_rate=mask_rate, lam=lam, max_iter=max_iter, learning_rate=learning_rate,
                       batch_size=batch_size),
            LayerParam(layers[2], inner_act=linear_func, act=linear_func, gnn_type=LayerParam.GAE,
                       mask_rate=mask_rate, lam=lam, max_iter=max_iter, learning_rate=learning_rate,
                       batch_size=batch_size),
        ]

    # ========== overlook setting ==========
    overlook_rates = None

    sgae = StackedGNN(features, adjacency, layers,
                      overlooked_rates=overlook_rates, BP_count=BP_count,
                      eta=eta, device=device,
                      labels=labels, metric_func=utils.clustering)

    utils.print_SGNN_info(sgae)

    print('============ Start Training ============')
    embedding = sgae.run()
    print('============ End Training ============')

    utils.print_SGNN_info(sgae)

    # ========== Clustering ==========
    print('============ Start Clustering ============')
    accuracy, nmi = utils.clustering_tensor(embedding.detach(), labels, relaxed_kmeans=True)
    finish_time = datetime.now()
    print(start_time.strftime("Process started at: " + "%Y-%m-%d %H:%M:%S"))
    print(finish_time.strftime("Process finished at: " + "%Y-%m-%d %H:%M:%S"))
    time_difference = finish_time - start_time

    # Extract hours, minutes, and seconds
    total_seconds = int(time_difference.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Training lasted {hours} hours, {minutes} minutes, {seconds} seconds")

    total_iterations = max_iter * len(layers) * ((BP_count * 2) + 1)
    print(f"Total iterations: {total_iterations}")
    efficiency = total_seconds / total_iterations
    print(f"Official efficiency: {efficiency}")

    return accuracy, efficiency, nmi, dataset_name
