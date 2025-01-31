import datetime
import logging
import sys
import os
import yaml

from src.tools import utils


# Load settings
def load_parameters(
        main_params_path: str = 'config/main.yaml',
        internal_params_path: str = 'config/settings.yaml') -> dict:
    """
    Read parameters and load them, both main ones and internal ones.

    Args:
        main_params_path: Path of file containing main parameters.
        internal_params_path: Path of file containing internal parameters.

    Returns:
        Object with all parameters.
    """
    with open(main_params_path, encoding='utf8') as par_file:
        main_params = yaml.safe_load(par_file)
    with open(internal_params_path, encoding='utf8') as par_file:
        internal_params = yaml.safe_load(par_file)
    params = utils.perform_dict_union_recursively(main_params, internal_params)
    params['global']['exec_name'] = datetime.datetime.now().strftime(
        '%Y%m%d_%H%M%S')
    return params

AWS_DEPLOYMENT = bool(os.environ.get("AWS_DEPLOYMENT", False))

# Set logger
params = load_parameters()
log_params = params['global']['logging']
logger = logging.getLogger(__name__)
hdlr_out = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(log_params['formatter']['format'],
                              log_params['formatter']['time_format'])
hdlr_out.setFormatter(formatter)
logger.addHandler(hdlr_out)
if log_params['file'] and not AWS_DEPLOYMENT:
    log_filename = log_params['file'].replace(
        '{exec_name}', params['global']['exec_name'])
    hdlr_file = logging.FileHandler(log_filename)
    hdlr_file.setFormatter(formatter)
    logger.addHandler(hdlr_file)
logger.setLevel(getattr(logging, log_params['level']))
logger.propagate = False

logger.info('Logger initialized')
