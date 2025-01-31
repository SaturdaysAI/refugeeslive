import importlib
import os
import re


def ensure_input_list(obj: object) -> list:
    """
    Ensure that input is a list; if not, return one.

    Args:
        obj: Any object.

    Returns:
        lst: Returned list.

    """
    if isinstance(obj, list):
        lst = obj
    elif not obj:
        lst = []
    else:
        lst = [obj]
    return lst


def import_pipeline(
        pipeline_name: str, module: str = 'src.pipelines') -> callable:
    """
    Given a pipeline module name, this function imports and returns the
    imported pipeline callable.

    Args:
        pipeline_name (str): pipeline module name to import.
        module (optional, str): base pipelines module. Default value is
            src.pipelines.

    Returns:
        (callable): pipeline module.
    """
    return importlib.import_module(f'{module}.{pipeline_name}')


def perform_dict_union_recursively(
        dict_x: dict, dict_y: dict) -> dict:
    """
    Perform a union of dictionaries, which means to take keys and values from
    both of them, without doing any replacement.
    This assumes that there are no equal low-level keys between both objects.
    The way it works is pretty simple:
    - It creates a new dictionary with unique keys from both inputs
    - Recursively call the function for common keys (which can be considered
      new dictionaries with their own keys)

    Args:
        dict_x: One of the dictionaries to merge.
        dict_y: The other dictionary to merge.

    Returns:
        Pure union of input dictionaries.

    Raises:
        AttributeError in case inputs share a key which is not another dict.
    """
    output = {}
    x_keys, y_keys = set(dict_x.keys()), set(dict_y.keys())
    shared_keys = list(x_keys.intersection(y_keys))
    unique_x = list(x_keys.difference(set(shared_keys)))
    unique_y = list(y_keys.difference(set(shared_keys)))
    for key in unique_x:
        output[key] = dict_x[key]
    for key in unique_y:
        output[key] = dict_y[key]
    for key in shared_keys:
        if not isinstance(dict_x[key], dict) \
           or not isinstance(dict_y[key], dict):
            raise AttributeError(
                f'Pure union cannot be made due to key ["{key}"]. '
                'Please review both dictionaries.')
        else:
            output[key] = perform_dict_union_recursively(
                dict_x[key], dict_y[key])
    return output


def ensure_unique_execution(path: str, version: str) -> None:
    """
    Ensure that execution is unique; this means that there are no results with
    the same directory name.

    Args:
        path: Directory where executions are stored.
        version: Name of version.

    Raises:
        Exception if execution results already exist.
    """
    results = os.path.join(path, version, "answers.csv")
    if os.path.exists(results):
        raise Exception(
            f'Execution {version} already exists in {path}.\n'
            'Please rename model version or delete previous results.')


def detect_invalid_and_replace(
        answer: str, invalid_regexp: str = '^(?s:.*?)[aA-zZ](?s:.*?)$',
        replacement: str = 'Invalid answer') -> str:
    """
    Detect if an answer is invalid.
    In this case, replace it to a valid answer.

    Args:
        answer: Invalid answer, not containing letters.
        invalid_regexp: Regular expression to detect whether answer is invalid
            or not
        replacement: Default replacement.

    Returns:
        Corrected answer in case it is invalid; else, provided answer.

    Note: Default invalid scenario consists of not containing any letters.
    """
    if not re.match(invalid_regexp, answer):
        answer = replacement
    return answer


def retrieve_module_and_callable(input_string: str) -> [object, str]:
    """
    Retrieve module and callable from a string. It can be either a function or
    a class method.

    Args:
        input_string: String containing path to callable.
    """
    library = '.'.join(input_string.split('.')[:-1])
    callable_ = input_string.split('.')[-1]
    try:
        # If it succeeds, it is indeed a function
        module = importlib.import_module(library)
    except ModuleNotFoundError:
        # In this case, it should be a class method
        library = '.'.join(input_string.split('.')[:-2])
        callable_ = input_string.split('.')[-2:]
        try:
            module = importlib.import_module(library)
        except ModuleNotFoundError:
            raise Exception(
                f'Function or class method provided not valid: {input_string}')
    return module, callable_
