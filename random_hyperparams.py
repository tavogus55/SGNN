import json
import random

def sample_hyperparams(filename, dataset_name):
    """Reads hyperparameter ranges from a JSON file and randomly selects a configuration."""

    random.seed()

    with open(filename, "r") as f:
        data = json.load(f)

    params = data[dataset_name]

    # Randomly sample values for global hyperparameters
    sampled_params = {
        "eta": random.choice(params["eta"]),
        "BP_count": random.choice(params["BP_count"]),
        "lam": random.choice(params["lam"]),
        "layers": []
    }

    # Sample values for each layer
    for layer in params["layers"]:
        sampled_layer = {
            "neurons": random.choice(layer["neurons"]),
            "inner_act": random.choice(layer["inner_act"]),
            "activation": random.choice(layer["activation"]),
            "learning_rate": random.choice(layer["learning_rate"]),
            "order": random.choice(layer["order"]),
            "max_iter": random.choice(layer["max_iter"]),
            "batch_size": random.choice(layer["batch_size"])
        }
        sampled_params["layers"].append(sampled_layer)

    return sampled_params

if __name__ == "__main__":
    random_hyperparams = sample_hyperparams()
    print(json.dumps(random_hyperparams, indent=4))
