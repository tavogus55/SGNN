from SGNN_tasks import run_classificaton, run_clustering
import json
from utils import sample_hyperparams, set_arg_parser, CustomFormatter
import logging
import torch

def run_experiment(cuda_num, exp_times, config, dataset_decision, tuning_file=None, logger=None):
    accuracy_list = []
    efficiency_list = []
    nmi_list = []
    total_accuracy = 0
    total_efficiency = 0
    total_nmi = 0

    for time in range(exp_times):
        accuracy = 0
        efficiency = 0
        nmi = 0
        logger.info(f"Running experiment {time + 1} of {exp_times}")
        if task_type == 'Clustering':
            accuracy, efficiency, nmi, dataset_name = run_clustering(dataset_decision, config)
            accuracy_list.append(accuracy)
            efficiency_list.append(efficiency)
            nmi_list.append(nmi)
        elif task_type == 'Classification':
            nmi = 0
            accuracy, efficiency, dataset_name = run_classificaton(cuda_num, dataset_decision, config, logger=logger)
            accuracy_list.append(accuracy)
            efficiency_list.append(efficiency)

        total_accuracy = total_accuracy + accuracy
        total_efficiency = total_efficiency + efficiency
        total_nmi = total_nmi + nmi

    average_accuracy = total_accuracy / exp_times
    average_efficiency = total_efficiency / exp_times
    average_nmi = total_nmi / exp_times
    logger.info(f"Experiment results")
    logger.info(f'Dataset used: {dataset_name}')
    logger.info(f'Task type: {task_type}')
    logger.info(f'Experiment count: {exp_times}')
    logger.info(f"All the accuracies: {accuracy_list}")
    logger.info(f"All the efficiencies: {efficiency_list}")
    logger.info(f"All the nmi: {nmi_list}")
    logger.info(f"The average accuracy is: {average_accuracy}")
    logger.info(f"The average efficiency is: {average_efficiency}")
    logger.info(f"The average nmi is: {average_nmi}")
    if isTuning is not None:
        tuning_file.write(f"All the accuracies: {accuracy_list}")
        tuning_file.write(f"All the efficiency: {efficiency_list}")

    return average_accuracy, average_efficiency, average_nmi


def main(cuda_num, dataset_decision, task_type, exp_times, isTuning, logger=None):

    if isTuning is None:
        with open('./config.json', 'r') as file:
            settings = json.load(file)
            config = settings[task_type][dataset_decision]
        run_experiment(cuda_num, exp_times, config, dataset_decision, logger=logger)
    else:
        f = open(f"./logs/tuning/tuning_{dataset_decision}_for_{isTuning}_times.txt", "a")
        tuning_accuracy_list = []
        tuning_efficiency_list = []
        for time in range(isTuning):
            logger.info(f"\n=======\nRunning hyperparameter tuning {time + 1} of {isTuning}\n=======")
            config = sample_hyperparams("ranges.json", dataset_decision)
            logger.info(json.dumps(config, indent=4))
            f.write(json.dumps(config, indent=4))
            average_accuracy, average_efficiency, average_nmi = run_experiment(cuda_num, exp_times, config,
                                                                               dataset_decision, f, logger=logger)
            tuning_accuracy_list.append(average_accuracy)
            tuning_efficiency_list.append(average_efficiency)
        logger.info(f"All the tuning accuracies: {tuning_accuracy_list}")
        logger.info(f"All the tuning efficiencies: {tuning_efficiency_list}")


if __name__ == "__main__":
    cuda_num, dataset_decision, task_type, exp_times, logPath, isTuning = set_arg_parser()

    # create logger with 'spam_application'
    logger = logging.getLogger("SGNN")
    logging.basicConfig(
        filename=logPath,
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    ch.setFormatter(CustomFormatter())

    logger.addHandler(ch)

    logger.info(f"Using the following arguments:\nCuda device number: {cuda_num}\nDataset: {dataset_decision}"
                f"\nTask type: {task_type}\nExperiments number: {exp_times}\nTuning: {isTuning}\n")

    logger.info(f"CUDA version: {torch.version.cuda}")  # Check the CUDA version supported by PyTorch
    logger.info(f"CUDA active: {torch.cuda.is_available()}")  # Check if CUDA is detected
    logger.info(f"Pytorch version: {torch.version.__version__}")  # Check PyTorch version

    main(cuda_num, dataset_decision, task_type, exp_times, isTuning, logger=logger)
