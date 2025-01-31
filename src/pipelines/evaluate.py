from src.data import extract, load
from src.models import evaluation
from src.tools import utils
from src.tools.startup import logger


def execute(settings: dict, global_settings: dict) -> None:
    """
    Evaluate generated chatbot answers in order to evaluate their overall
    quality. For this, a reference dataset is provided with the correct answer
    for each question. There are two supported types of evaluation:
    - Based on Semantic Answer Similarity (SAS)
    - Based on LLMs (LLM)
    The results will be appended to a dataframe containing all results.

    Args:
        settings: Execution specific settings.
        global_settings: Global settings.
    """

    logger.info('Ensure that execution is unique')
    chatbot_execution = settings['answers']['chatbot'].split('/')[-2]
    utils.ensure_unique_execution(
        settings['results']['details'], chatbot_execution + '.csv')

    logger.info('Evaluating execution: %s', chatbot_execution)

    logger.info('Read datasets')
    settings['answers']['reference'] = global_settings[
        'paths']['reference_dataset']
    datasets = extract.read_evaluation_datasets(settings['answers'])

    logger.info('Split dataset by language equality')
    df_diff, df_same = evaluation.split_by_language_equality(
        datasets)

    model_results = {}
    for evaluation_type in list(settings['models']):
        logger.info(f'Evaluate through {evaluation_type}')
        if evaluation_type not in settings['specifics']:
            settings['specifics'][evaluation_type] = {}
        model_results[evaluation_type] = evaluation.perform_evaluation(
            df_same, evaluation_type, settings['models'][evaluation_type],
            settings['specifics'][evaluation_type])

    logger.info('Join detailed results')
    df_details = evaluation.join_all_results(
        model_results, df_diff, datasets['reference'])

    logger.info('Generate aggregated results')
    results, grouped = evaluation.generate_aggregated_results(
        df_details, ['language', 'context'])
    logger.info(f'Results are: {results}')
    logger.info(f'Grouped results are:\n {grouped}')

    logger.info('Store results')
    results = {'execution': chatbot_execution} | results
    load.store_evaluation_results(df_details, results, settings['results'])

    logger.info('Generate stats')
    stats = evaluation.generate_stats(df_details)
    load.store_stats(stats, chatbot_execution, settings['results'])
