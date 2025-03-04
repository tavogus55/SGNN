from GNN_tasks import run_classificaton_with_SGNN, run_clustering_with_SGNN, run_classification_with_SGC
import json
from utils import sample_hyperparams, set_arg_parser, get_logger
import torch


def run_experiment(cuda_num, exp_times, config, dataset_decision, model_decision, logger=None):
    accuracy_list = []
    efficiency_list = []
    nmi_list = []
    time_taken_list = []
    total_accuracy = 0
    total_efficiency = 0
    total_time_taken = 0
    total_nmi = 0

    for time in range(exp_times):
        accuracy = 0
        efficiency = 0
        nmi = 0
        time_taken = 0
        logger.info(f"Running experiment {time + 1} of {exp_times}")
        if task_type == 'Clustering':
            if model_decision == 'SGNN':
                accuracy, efficiency, nmi, dataset_name = run_clustering_with_SGNN(dataset_decision, config)
            accuracy_list.append(accuracy)
            efficiency_list.append(efficiency)
            nmi_list.append(nmi)
        elif task_type == 'Classification':
            if model_decision == 'SGNN':
                accuracy, efficiency, time_taken = run_classificaton_with_SGNN(cuda_num, dataset_decision, config,
                                                                               logger=logger)
            elif model_decision == 'SGC':
                accuracy, efficiency, time_taken = run_classification_with_SGC(cuda_num, dataset_decision, config,
                                                                               logger=logger)
            else:
                exit()
            accuracy_list.append(accuracy)
            efficiency_list.append(efficiency)
            time_taken_list.append(time_taken)

        total_accuracy = total_accuracy + accuracy
        total_efficiency = total_efficiency + efficiency
        total_time_taken = total_time_taken + time_taken
        total_nmi = total_nmi + nmi

    average_accuracy = total_accuracy / exp_times
    average_efficiency = total_efficiency / exp_times
    average_time_taken = total_time_taken / exp_times
    average_nmi = total_nmi / exp_times
    logger.info(f"\nEXPERIMENT RESULTS")
    logger.info(f'Dataset used: {dataset_decision}')
    logger.info(f'Model used: {model_decision}')
    logger.info(f'Task type: {task_type}')
    logger.info(f'Experiment count: {exp_times}')
    logger.info(f"All the accuracies: {accuracy_list}")
    logger.info(f"All the efficiencies: {efficiency_list}")
    logger.info(f"All the times taken: {time_taken_list}")
    logger.info(f"All the nmi: {nmi_list}")
    logger.info(f"The average accuracy is: {average_accuracy}")
    logger.info(f"The average efficiency is: {average_efficiency}")
    logger.info(f"The average time taken is: {average_time_taken}")
    logger.info(f"The average nmi is: {average_nmi}")

    return average_accuracy, average_efficiency, average_nmi, average_time_taken


def main(cuda_num, dataset_decision, model_decision, task_type, exp_times, isTuning, logger=None):

    if isTuning is None:
        with open('./config.json', 'r') as file:
            settings = json.load(file)
            config = settings[model_decision][task_type][dataset_decision]
            logger.info(json.dumps(config, indent=4))
        run_experiment(cuda_num, exp_times, config, dataset_decision, model_decision, logger=logger)
    else:
        tuning_accuracy_list = []
        tuning_efficiency_list = []
        tuning_time_taken_list = []
        for time in range(isTuning):
            logger.info(f"\n=======\nRunning hyperparameter tuning {time + 1} of {isTuning}\n=======")
            config = sample_hyperparams("ranges.json", dataset_decision)
            logger.info(json.dumps(config, indent=4))
            average_accuracy, average_efficiency, average_nmi, average_time_taken = run_experiment(cuda_num, exp_times,
                                                                                                   config,
                                                                                                   dataset_decision,
                                                                                                   model_decision,
                                                                                                   logger=logger)
            tuning_accuracy_list.append(average_accuracy)
            tuning_efficiency_list.append(average_efficiency)
            tuning_time_taken_list.append(average_time_taken)
        logger.info(f"All the tuning accuracies: {tuning_accuracy_list}")
        logger.info(f"Best accuracy: {max(tuning_accuracy_list)}")
        logger.info(f"All the tuning efficiencies: {tuning_efficiency_list}")
        logger.info(f"Best efficiency: {min(tuning_efficiency_list)}")
        logger.info(f"All the times taken: {tuning_efficiency_list}")
        logger.info(f"Best time taken: {min(tuning_time_taken_list)}")


if __name__ == "__main__":
    cuda_num, dataset_decision, model_decision, task_type, exp_times, logPath, isTuning = set_arg_parser()
    if "local_1" in logPath:
        logPath = f"./logs/{model_decision}_{dataset_decision}.log"

    logger = get_logger(f"{model_decision}", logPath)

    logger.info(f"Dataset: {dataset_decision}")  # Check the CUDA version supported by PyTorch
    logger.info(f"Model: {model_decision}")  # Check the CUDA version supported by PyTorch
    logger.info(f"CUDA num: {cuda_num}")  # Check the CUDA version supported by PyTorch
    logger.info(f"Task: {task_type}")  # Check the CUDA version supported by PyTorch
    logger.info(f"Number of experiments: {exp_times}")  # Check the CUDA version supported by PyTorch
    logger.info(f"CUDA version: {torch.version.cuda}")  # Check the CUDA version supported by PyTorch
    logger.info(f"CUDA active: {torch.cuda.is_available()}")  # Check if CUDA is detected
    logger.info(f"Pytorch version: {torch.version.__version__}")  # Check PyTorch version

    main(cuda_num, dataset_decision, model_decision, task_type, exp_times, isTuning, logger=logger)
