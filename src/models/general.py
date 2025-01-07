import pandas as pd
from src.tools import utils


def generate_multiple_answers(
        questions: list[str], module: str, params: dict) -> pd.DataFrame:
    """
    Generate answers from multiple questions through the specified options.
    This involves retrieving the function to call depending on whether it
    corresponds to a function, or a class method:
    - In case it is a function, it is directly called with the specified
    parameters.
    - In case it is a class method, the class is first instantiated with the
    specified parameters, and then the method is called.

    Args:
        questions: Questions to ask.
        module: Module name invoking the chatbot for each question.
        params: Specific parameters for the module to consider.

    Returns:
        Dataset with questions and corresponding answers.
    """
    df_ref = pd.DataFrame({'question': questions})
    if not params:
        params = {}
    module, callable_ = utils.retrieve_module_and_callable(module)
    # Check whether it is a class method (callable is a list) or not
    if isinstance(callable_, list):
        class_obj = getattr(module, callable_[0])(params)
        func_to_call = getattr(class_obj, callable_[1])
        df_ref['answer'] = df_ref['question'].apply(
            lambda x: func_to_call(x))
    else:
        func_to_call = getattr(module, callable_)
        df_ref['answer'] = df_ref['question'].apply(
            lambda x: func_to_call(x, **params))
    return df_ref
