import copy
import re

import numpy as np
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from lingua import LanguageDetectorBuilder
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.tools.startup import logger

BASE_COLS = ['question', 'answer']
MAIN_COLS = ['question', 'reference', 'chatbot']


def split_by_language_equality(datasets: dict
                               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a dataset for comparing answers, and split it in two, depending on
    their language equality: one for answers' that are in the same language,
    and the other those that are not. This is done because answers in different
    languages, are finally automatically rated as 0.

    Args:
        datasets: Object containing both datasets for evaluation.
    """
    datasets = copy.deepcopy(datasets)
    answer_types = list(datasets)
    for dataset in list(datasets):
        datasets[dataset] = datasets[dataset][BASE_COLS]
    df_evaluation = datasets['reference'].merge(
        datasets['chatbot'], on='question')
    df_evaluation.columns = ['question', 'reference', 'chatbot']
    detector = LanguageDetectorBuilder.from_all_languages(
        ).with_preloaded_language_models().build()
    for answer_type in answer_types:
        df_evaluation['lang_' + answer_type] = df_evaluation[
            answer_type].apply(
                lambda x: detector.compute_language_confidence_values(x)[
                    0].language.name)
    df_evaluation['same_language'] = df_evaluation[
        'lang_reference'] == df_evaluation['lang_chatbot']
    df_diff = df_evaluation.loc[
        ~df_evaluation['same_language']].copy(deep=True)
    df_same = df_evaluation.loc[df_evaluation['same_language']]
    return df_diff, df_same


def perform_evaluation(
        df_model: pd.DataFrame, type_: str, models: list,
        settings: dict) -> dict:
    """
    Evaluate chatbot responses independently of method used.
    This works in the following way, for each model:
    1. A dataset is created containing reference answer and chatbot's
    2. The specific metric value is computed for each pair of answers
    3. Take the mean value for all metric values

    Args:
        datasets: Object containing both datasets for evaluation.
            Requires both keys: 'reference' and 'chatbot'.
        type_: Evaluation type.
        models: Models to be used, if any.
        settings: Specific evaluation settings.

    Returns:
        Object with evaluation results, with the following structure:
            - {type_model}: value
    """
    df_model = df_model[MAIN_COLS].copy(deep=True)
    eval_function_name = 'evaluate_answers_via_' + type_.lower()
    eval_function = globals()[eval_function_name]
    for model in models:
        logger.info(f'-> Model used: {model}')
        value_name = type_ + '_' + model
        df_model[value_name] = df_model.apply(lambda row: eval_function(
                row['reference'], row['chatbot'], model, **settings), axis=1)
        df_model[value_name] = df_model[value_name].clip(0, 1)
    return df_model


def evaluate_answers_via_sas(reference: str, chatbot: str, model: str) -> int:
    """
    Evaluate chatbot response via Semantic Analysis Similarity (SAS) metric.
    The way this is done is, given an LLM model:
    1. Generate the text representation (embeddings) for both the reference
       answers and the chatbot ones.
    2. Compute the cosine similarity between each pair of embeddings

    Args:
        reference: Reference answer.
        chatbot: Chatbot answer.
        model: Model name to use.

    Returns:
        Evaluation metric computed through SAS method.
    """
    model_object = SentenceTransformer(model)
    answers = (reference, chatbot)
    answers_embeddings = ()
    for answer in list(answers):
        answer_embeddings = np.array(
            model_object.encode(answer)).reshape(1, -1)
        answers_embeddings = answers_embeddings + (answer_embeddings,)
    metric = np.diag(
        cosine_similarity(answers_embeddings[0], answers_embeddings[1]))[0]
    return metric


def evaluate_answers_via_llm(
        reference: str, chatbot: str, model: str,
        prompt_filename: str, max_tokens: int, default_value: float) -> int:
    """
    Evaluate chatbot response via LLM-powered metric.
    The way this is done is, given an LLM model:
    1. Get the prompt template to be used to help with the ranking
    2. Ask the LLM model about the metric value

    Args:
        reference: Reference answer.
        chatbot: Chatbot answer.
        model: Model name to use.
        prompt_filename: File name containing prompt template.
        max_tokens: Maximum tokens to generate by the model.
        default_value: Default metric value if not provided.

    Returns:
        Evaluation metric computed through LLM method.
    """
    model_obj = HuggingFacePipeline.from_model_id(
        model_id=model, task="text-generation",
        pipeline_kwargs={"max_new_tokens": max_tokens})
    with open(prompt_filename, 'r') as file:
        prompt = file.read()
    prompt_object = PromptTemplate.from_template(prompt)
    chain = prompt_object | model_obj
    result = chain.invoke({'reference': reference, 'chatbot': chatbot})
    result_trimmed = result.replace('\n', ' ')
    regexp_float = r'^.*?(0[^\.]|1|[0-9]\.[0-9]+).*$'
    try:
        metric = float(re.match(regexp_float, result_trimmed)[1])
    except TypeError:
        metric = default_value
    return metric


def join_all_results(
        model_results: dict, df_diff: pd.DataFrame,
        df_reference: pd.DataFrame) -> pd.DataFrame:
    """
    Join all detailed results from evaluation to generate a single dataset.
    This is done by iterating through evaluation types. For each one:
    - Results from different language are appended to all model results.
    - Their model results are added to a global dataset with detailed results.

    Args:
        model_results: Results from models.
        df_diff: Results from different languages
        df_reference: Reference dataset.

    Returns:
        Dataset with detailed results.
    """
    df_details = model_results[list(model_results)[0]][MAIN_COLS]
    df_details = pd.concat([df_details, df_diff[MAIN_COLS]])
    for type_ in list(model_results):
        df_type = model_results[type_]
        df_details = df_details.merge(df_type, on=MAIN_COLS, how='left')
    df_details = df_details.fillna(0)
    df_details = df_details.merge(
        df_reference.drop(columns='answer'), on=['question'], how='left')
    return df_details


def generate_aggregated_results(
        df_details: pd.DataFrame, groups: list) -> tuple[dict, dict]:
    """
    Generate aggregated results from detailed ones, which are question by
    question.
    The aggregation uses the mean to get a single value for all models tested.
    There are two types of results:
    - Global
    - By group

    Args:
        df_details: Dataset containing detailed results.
        groups: Groups for which to compute aggregated metrics.

    Returns:
        - Global results
        - Grouped results
    """
    not_value_cols = MAIN_COLS + groups
    value_cols = list(set(list(df_details)).difference(set(not_value_cols)))
    # If metric is only one, it can be named "metric" to simplify the output
    if len(value_cols) == 1:
        df_details = df_details.rename(columns={value_cols[0]: 'metric'})
        value_cols = ['metric']
    result_values = df_details[value_cols].mean()
    results = dict(zip(value_cols, [x.__round__(3) for x in result_values]))
    grouped = df_details.groupby(groups)[
        value_cols].mean().round(3).reset_index().to_dict('records')
    return results, grouped


def generate_stats(df_details: pd.DataFrame) -> pd.DataFrame:
    """
    Generate aggregated results stats.

    Args:
        df_details: Dataset containing detailed results.

    Returns:
        Aggregated statistics of the results.
    """
    df = df_details.rename(columns={
        "SAS_sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 
        "similarity"})
    df['in_context'] = df['chatbot'].apply(
        lambda x: False 
            if x == 'Disculpa, no dispongo de suficientes datos para contestar a esta pregunta.'
            or x == "Sorry, I don't have enough data to answer this question."
            else True)
    stats = df[["similarity", "context", "in_context", "language"]]\
        .groupby(["context", "in_context", "language"]).agg({"mean", len})
    stats = stats.reset_index()
    return stats
