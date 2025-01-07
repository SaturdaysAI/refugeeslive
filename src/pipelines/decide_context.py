import pandas as pd
from src.models import general
from src.tools.startup import logger


def execute(settings: dict, global_settings: dict) -> None:
    """
    For each question in the reference dataset, executes the context detection
    modules to determine whether a question belongs to the context or not.
    Answers are exported to a CSV file. 

    Args:
        settings: Execution specific settings.
        global_settings: Global settings.
    """
    if not settings['version_name']:
        settings['version_name'] = global_settings['exec_name']
    
    logger.info("Current execution: %s", settings['version_name'])

    logger.info('Get evaluation questions')
    df_questions = pd.read_csv(
        global_settings['paths']['reference_dataset'], delimiter='|')
    questions = list(df_questions['question'])

    logger.info('decide contexts')
    df_answers = general.generate_multiple_answers(
        questions, settings['module'], settings['params'])

    df_questions = df_questions.drop(columns=['answer'])
    df_answers = df_questions.merge(df_answers, on='question')
    df_answers["count"] = 1

    # process results
    stats = df_answers[["context", "answer", "language", "count"]]\
        .groupby(["context", "answer", "language"]).sum().reset_index()

    logger.info('Store results')
    path = settings['store_path']
    version = settings['version_name']
    import os
    results_dir = os.path.join(path, version)
    os.makedirs(results_dir, exist_ok=True)
    # Export answers
    filepath = os.path.join(results_dir, 'decide_context.csv')
    stats.to_csv(filepath, index=None)
