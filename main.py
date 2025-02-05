from run_classification import run_classificaton
from run import run_clustering
import argparse
import json
from random_hyperparams import sample_hyperparams

def run_experiment(exp_times, config, dataset_decision, tuning_file=None):

    for time in range(exp_times):
        accuracy_list = []
        efficiency_list = []
        nmi_list = []
        total_accuracy = 0
        total_efficiency = 0
        total_nmi = 0
        accuracy = 0
        efficiency = 0
        nmi = 0
        print('========================')
        print('========================')
        print(f"Running experiment {time + 1} of {exp_times}")
        print('========================')
        print('========================')
        if task_type == 'Clustering':
            accuracy, efficiency, nmi, dataset_name = run_clustering(dataset_decision)
            accuracy_list.append(accuracy)
            efficiency_list.append(efficiency)
            nmi_list.append(nmi)
        elif task_type == 'Classification':
            nmi = 0
            accuracy, efficiency, dataset_name = run_classificaton(dataset_decision, config)
            accuracy_list.append(accuracy)
            efficiency_list.append(efficiency)

        total_accuracy = total_accuracy + accuracy
        total_efficiency = total_efficiency + efficiency
        total_nmi = total_nmi + nmi

        average_accuracy = total_accuracy / exp_times
        average_efficiency = total_efficiency / exp_times
        average_nmi = total_nmi / exp_times
        print('========================')
        print('========================')
        print(f"Experiment results")
        print('========================')
        print('========================')
        print(f'Dataset used: {dataset_name}')
        print(f'Task type: {task_type}')
        print(f'Experiment count: {exp_times}')
        print(f"All the accuracies: {accuracy_list}")
        print(f"All the efficiencies: {efficiency_list}")
        print(f"All the nmi: {nmi_list}")
        print(f"The average accuracy is: {average_accuracy}")
        print(f"The average efficiency is: {average_efficiency}")
        print(f"The average nmi is: {average_nmi}")
        if isTuning is not None:
            tuning_file.write(f"All the accuracies: {accuracy_list}")
            tuning_file.write(f"All the efficiency: {efficiency_list}")

        return average_accuracy, average_efficiency, average_nmi


def main(dataset_decision, task_type, exp_times, isTuning):

    if isTuning is None:
        with open('config.json', 'r') as file:
            settings = json.load(file)
            config = settings[task_type][dataset_decision]
        run_experiment(exp_times, config, dataset_decision)
    else:
        f = open(f"tuning_{dataset_decision}_for_{isTuning}_times.txt", "a")
        tuning_accuracy_list = []
        tuning_efficiency_list = []
        for time in range(isTuning):
            print('========================')
            print('========================')
            print(f"Running hyperparameter tuning {time + 1} of {isTuning}")
            print('========================')
            print('========================')
            config = sample_hyperparams("ranges.json", dataset_decision)
            print(json.dumps(config, indent=4))
            f.write(json.dumps(config, indent=4))
            average_accuracy, average_efficiency, average_nmi = run_experiment(exp_times, config, dataset_decision, f)
            tuning_accuracy_list.append(average_accuracy)
            tuning_efficiency_list.append(average_efficiency)
        print('========================')
        print('========================')
        print(f"All the tuning accuracies: {tuning_accuracy_list}")
        print(f"All the tuning efficiencies: {tuning_efficiency_list}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script")
    parser.add_argument("--name", type=str, required=True, help="Dataset name ()")
    parser.add_argument("--task", type=str, required=True, help="Classification or Clustering")
    parser.add_argument("--exp", type=int, required=True, help="How many times do you want to run the exercise")
    parser.add_argument("--tuning", type=int, help="How many times you want to tune the hyperparameters")
    args = parser.parse_args()

    dataset_decision = args.name
    task_type = args.task
    exp_times = args.exp
    isTuning = args.tuning

    main(dataset_decision, task_type, exp_times, isTuning)