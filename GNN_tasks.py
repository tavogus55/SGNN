from torch.fx.passes.infra.partitioner import logger
from model.SGC import SGC, train, evaluate
from model.SGNN import *
from data_loader import get_training_data
import warnings
from datetime import datetime
from torch_geometric.loader import NeighborLoader

warnings.filterwarnings('ignore')
utils.set_seed(0)


def run_classificaton_with_SGNN(cuda_num, dataset_choice, config, logger=None):

    start_time = datetime.now()

    data = get_training_data(dataset_choice)

    labels = data.y
    features = data.x
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    adjacency = data.adjacency

    n_class = np.unique(labels).shape[0]
    if (type(features) is not np.ndarray and dataset_choice != "Reddit" and dataset_choice != "Mag"
            and dataset_choice != "Yelp"):
        features = features.todense()
    features = torch.Tensor(features)
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")

    # ========== training setting ==========
    eta = config["eta"]
    BP_count = config["BP_count"]
    is_large = config["isLarge"]

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

        if is_large:
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
                                labels=labels, metric_func=utils.classification, logger=logger)

    utils.print_SGNN_info(sgnn, logger=logger)

    logger.debug('============ Start Training ============')
    prediction = sgnn.run()
    logger.debug('============ End Training ============')

    # ========== Testing ==========
    logger.info('============ Start testing ============')
    logger.info("Training accuracy")
    utils.classification(prediction, labels, train_mask, logger=logger)
    logger.info("Validation accuracy")
    utils.classification(prediction, labels, val_mask, logger=logger)
    logger.info("Test accuracy")
    accuracy = utils.classification(prediction, labels, test_mask, logger=logger)
    finish_time = datetime.now()

    time_difference = finish_time - start_time

    # Extract hours, minutes, and seconds
    total_seconds = int(time_difference.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info(start_time.strftime("Process started at: " + "%Y-%m-%d %H:%M:%S"))
    logger.info(finish_time.strftime("Process started at: " + "%Y-%m-%d %H:%M:%S"))
    logger.info(f"Training lasted {hours} hours, {minutes} minutes, {seconds} seconds")
    total_max_iter = 0
    for layer in layer_config:
        total_max_iter = total_max_iter + layer["max_iter"]

    total_iterations = total_max_iter*((BP_count*2)+1)
    logger.info(f"Total iterations: {total_iterations}")
    efficiency = total_seconds / total_iterations
    logger.info(f"Official efficiency: {efficiency}")

    return accuracy, efficiency, total_seconds


def run_classification_with_SGC(cuda_num, dataset_choice, config, logger=None):

    start_time = datetime.now()
    is_large = config["isLarge"]
    epochs = config["epochs"]
    batch_size_train = None
    batch_size_test = None
    if is_large:
        batch_size_train = config["batch_size_train"]
        batch_size_test = config["batch_size_test"]
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]

    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    data = get_training_data(dataset_choice)

    if is_large:
        train_loader = NeighborLoader(
            data,
            num_neighbors=[10, 10],  # Sample 10 neighbors per layer
            batch_size=batch_size_train,
            input_nodes=data.train_mask
        )

        test_loader = NeighborLoader(
            data,
            num_neighbors=[10, 10],
            batch_size=batch_size_test,
            input_nodes=data.test_mask
        )
    else:
        train_loader = None
        test_loader = None

    num_features = data.num_node_features
    num_classes = data.num_classes

    model = SGC(num_features, num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(1, epochs):
        loss = train(model, optimizer, device, train_loader, dataset_name=dataset_choice)
        logger.debug(f'Epoch {epoch}: Loss: {loss:.4f}')

    accuracy = evaluate(model, device, test_loader, dataset_name=dataset_choice)
    logger.info(f"Test Accuracy: {accuracy:.4f}")

    finish_time = datetime.now()
    time_difference = finish_time - start_time

    logger.info(start_time.strftime("Process started at: %Y-%m-%d %H:%M:%S"))
    logger.info(finish_time.strftime("Process finished at: %Y-%m-%d %H:%M:%S"))

    total_seconds = int(time_difference.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info(f"Training lasted {hours} hours, {minutes} minutes, {seconds} seconds")

    total_iterations = epochs
    efficiency = total_seconds / total_iterations
    logger.info(f"Official efficiency: {efficiency}")

    return accuracy, efficiency, total_seconds


def run_clustering_with_SGNN(dataset_choice, config):
    dataset_name = None
    start_time = datetime.now()

    adjacency, features, labels, train_mask, val_mask, test_mask  = get_training_data(dataset_choice)

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
                      labels=labels, metric_func=utils.clustering, logger=logger)

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