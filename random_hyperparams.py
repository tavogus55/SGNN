import json
import random

def sample_hyperparams(filename, dataset_name):
    """Reads hyperparameter ranges from a JSON file and randomly selects a configuration."""
    random.seed()

    with open(filename, "r") as f:
        data = json.load(f)

    params = data["Test"]

    # Randomly sample values for global hyperparameters
    sampled_params = {
        "eta": random.choice(params["eta"]),
        "BP_count": random.choice(params["BP_count"]),
        "lam": random.choice(params["lam"]),
        "layers": []
    }

    # Determine random number of layers (2 or 3)
    num_layers = random.choice([2, 3])

    # Sample values for each layer dynamically
    for _ in range(num_layers):
        sampled_layer = {
            "neurons": random.choice(params["layer"][0]["neurons"]),
            "inner_act": random.choice(params["layer"][0]["inner_act"]),
            "activation": random.choice(params["layer"][0]["activation"]),
            "learning_rate": random.choice(params["layer"][0]["learning_rate"]),
            "order": random.choice(params["layer"][0]["order"]),
            "max_iter": random.choice(params["layer"][0]["max_iter"]),
            "batch_size": random.choice(params["layer"][0]["batch_size"])
        }
        sampled_params["layers"].append(sampled_layer)

    return sampled_params

if __name__ == "__main__":
    random_hyperparams = sample_hyperparams()
    print(json.dumps(random_hyperparams, indent=4))
