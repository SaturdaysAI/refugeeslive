import json
import os
import shutil

import pandas as pd
import yaml
import torch


def store_evaluation_results(
        df_details: pd.DataFrame, results: dict, settings: dict) -> None:
    """
    Store both detailed evaluation results and global results.
    For detailed results, a new file is created within the specified path.
    For global results; it works in the following way:
    - If the file exists, read all results; if not, assume empty ones
    - Append current results to all of them
    - Store results to file again

    Args:
        df_details (pd.DataFrame): Dataset with detailed results.
        results (dict): Global evaluation results.
        settings (dict): Store settings. It should contain the following keys:
            - global: A json file path
            - details: A directory path
    """
    # Store detailed results
    if not os.path.exists(settings['details']):
        os.makedirs(settings['details'], exist_ok=True)
    details_filename = os.path.join(
        settings['details'], results['execution'] + '.csv')
    df_details.to_csv(details_filename, index=None)
    # Append global results
    if not os.path.exists(settings['global']):
        all_results = []
    else:
        with open(settings['global'], 'r') as file:
            all_results = json.load(file)
    all_results.append(results)
    with open(settings['global'], 'w') as file:
        json.dump(all_results, file)


def store_answers_results(
        df_answers: pd.DataFrame, settings: dict,
        path: str = 'store_path', version: str = 'version_name') -> None:
    """
    Store answer generation results. This means storing the following:
    - Dataset with questions and related answers generated by the chatbot.
    - Configuration used for generating the answers.

    Args:
        df_answers (pd.DataFrame): Dataset with reference questions and chatbot
        answers'.
        settings (dict): Model parameters.
        path (str): Key within settings with directory where executions are
        stored.
        version (str): Key within settings with name of version.
    """
    path = settings.pop(path)
    version = settings.pop(version)
    results_dir = os.path.join(path, version)
    os.makedirs(results_dir, exist_ok=True)
    # Export answers
    filepath = os.path.join(results_dir, 'answers.csv')
    df_answers.to_csv(filepath, index=None, sep='|')
    # Export parameters
    filepath = os.path.join(results_dir, 'settings.yaml')

    # Define a custom representer for torch.float16
    def torch_dtype_representer(dumper, data):
        return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))

    # Register the custom representer
    yaml.add_representer(torch.dtype, torch_dtype_representer)

    with open(filepath, encoding='utf8', mode='w') as file:
        yaml.dump(settings, file)
    # Export templates
    templates = ['generation', 'context']
    for template in templates:
        if template in settings['params']:
            filepath = os.path.join(results_dir, template + '.txt')
            prompt_file = settings['params'][template]
            shutil.copyfile(prompt_file, filepath)


def store_stats(stats: pd.DataFrame, chatbot_execution: str, settings: dict
                ) -> None:
    """
    Store result stats dataframe

    Args:
        stats (pd.DataFrame): Dataset with execution metrics statistics.
        chatbot_execution (str): name of the execution.
        settings (dict): settings used in the chatbot execution.
    """
    if not os.path.exists(settings['stats']):
        os.makedirs(settings['stats'], exist_ok=True)
    stats_filename = os.path.join(
        settings['stats'], 'stats_' + chatbot_execution + '.csv')
    stats.to_csv(stats_filename, index=None)
    
def save_json_file(content: dict, filepath: str) -> None:
    """
    Saves a dictionary as a JSON file at the specified filepath.

    Args:
        content (dict): The dictionary to save in JSON format.
        filepath (str): The path where the JSON file will be saved.
    """
    json.dump(content, open(filepath, "w"))
