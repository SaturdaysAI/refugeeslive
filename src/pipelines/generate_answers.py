import pandas as pd
from src.data import load
from src.models import general
from src.tools import utils
from src.tools.startup import logger


def execute(settings: dict, global_settings: dict) -> None:
    """
    Generate chatbot answers in order to later on compute overall quality of
    aforementioned chatbot.

    Args:
        settings: Execution specific settings.
        global_settings: Global settings.
    """
    if not settings['version_name']:
        settings['version_name'] = global_settings['exec_name']

    logger.info('Ensure that execution is unique')
    utils.ensure_unique_execution(
        settings['store_path'], settings['version_name'])
    
    logger.info("Current execution: %s", settings['version_name'])

    logger.info('Get evaluation questions')
    df_questions = pd.read_csv(
        global_settings['paths']['reference_dataset'], delimiter='|')
    questions = list(df_questions['question'])

    logger.info('Generate answers')
    df_answers = general.generate_multiple_answers(
        questions, settings['module'], settings['params'])

    logger.info('Store results')
    load.store_answers_results(df_answers, settings)
