from torch_geometric.datasets import (Planetoid, Reddit, Flickr, FacebookPagePage, Actor, LastFMAsia, DeezerEurope,
                                      Amazon)

# Define available datasets
available_datasets = ['Cora', 'CiteSeer', 'PubMed', 'Reddit', 'Flickr', 'FacebookPagePage', 'Actor',
                      'LastFMAsia', 'DeezerEurope', 'Amazon']

# Prompt the user to select a dataset
print("Available datasets:")
for i, dataset_name in enumerate(available_datasets, start=1):
    print(f"{i}: {dataset_name}")

choice = int(input("Enter the number corresponding to the dataset you want to download: "))

# Validate the choice
if 1 <= choice <= len(available_datasets):
    selected_dataset = available_datasets[choice - 1]
    print(f"Downloading the {selected_dataset} dataset...")

    # Download the selected dataset
    if selected_dataset == 'Reddit':
        dataset = Reddit(root='data/reddite')
    elif selected_dataset == 'Flickr':
        dataset = Flickr(root='data/Flickr')
    elif selected_dataset == 'FacebookPagePage':
        dataset = FacebookPagePage(root='data/FacebookPagePage')
    elif selected_dataset == 'Actor':
        dataset = Actor(root='data/Actor')
    elif selected_dataset == 'LastFMAsia':
        dataset = LastFMAsia(root='data/LastFMAsia')
    elif selected_dataset == 'DeezerEurope':
        dataset = DeezerEurope(root='data/DeezerEurope')
    elif selected_dataset == 'Amazon':
        dataset = Amazon(root='data/Amazon', name='Photo')
    else:
        dataset = Planetoid(root='data/', name=selected_dataset)

    print(f"The {selected_dataset} dataset has been downloaded successfully!")
else:
    print("Invalid choice. Please run the script again and select a valid dataset.")
