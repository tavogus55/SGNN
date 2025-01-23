from run_classification import run_classificaton
from run import run_clustering


def main():

    dataset_decision = input("Choose which dataset to use\n1. Cora\n2. Citeseer\n3. Pubmed\n4. Flickr"
                             "\n5. FacebookPagePage" "\n6. Actor\n7. LastFMAsia\n8. DeezerEurope"
                             "\n9. Amazon Computers\n10. Amazon Photos\n\nYour input: ")

    task_decision = input(
        "Choose which dataset to use\n1. Clustering\n2. Classification\n\nYour input: ")

    exp_times = int(input("How many times do you want to run the experiment?"))

    average_accuracy = 0
    average_efficiency = 0
    total_accuracy = 0
    total_efficiency = 0

    accuracy_list = []
    efficiency_list = []

    for time in range(exp_times):
        print('========================')
        print('========================')
        print(f"Running experiment {time}")
        print('========================')
        print('========================')
        if task_decision == "1":
            task_name = 'Clustering'
            accuracy, efficiency, dataset_name = run_clustering(dataset_decision)
            accuracy_list.append(accuracy)
            efficiency_list.append(efficiency)
        elif task_decision == "2":
            task_name = 'Classification'
            accuracy, efficiency, dataset_name = run_classificaton(dataset_decision)
            accuracy_list.append(accuracy)
            efficiency_list.append(efficiency)


        total_accuracy = total_accuracy + accuracy
        total_efficiency = total_efficiency + efficiency

    average_accuracy = total_accuracy / exp_times
    average_efficiency = total_efficiency / exp_times
    print('========================')
    print('========================')
    print(f"Experiment results")
    print('========================')
    print('========================')
    print(f'Dataset used: {dataset_name}')
    print(f'Task type: {task_name}')
    print(f'Experiment count: {exp_times}')
    print(f"All the accuracies: {accuracy_list}")
    print(f"All the efficiencies: {efficiency_list}")
    print(f"The average accuracy is: {average_accuracy}")
    print(f"The average efficiency is: {average_efficiency}")


if __name__ == "__main__":
    main()