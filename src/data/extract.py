import pandas as pd


def read_evaluation_datasets(answers: dict) -> dict:
    """
    Read datasets that will be used for evaluation, which are the following:
    - reference: Dataset with correct answers
    - chatbot: Dataset with chatbot answers

    Args:
        answers: Specifies input file for each type of dataset.
            Requires both keys: 'reference' and 'chatbot'.

    Returns:
        Object containing both datasets for evaluation.
            Contains both keys: 'reference' and 'chatbot'.

    """
    datasets = {}
    for dataset in list(answers):
        datasets[dataset] = pd.read_csv(answers[dataset], delimiter='|')
    return datasets
