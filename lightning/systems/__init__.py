from .baseline import BaselineSystem
from .meta import MetaSystem
from .imaml import IMAMLSystem
from .xvec_fit import XvecFitSystem

SYSTEM = {
    "meta": MetaSystem,
    "imaml": IMAMLSystem,
    "baseline": BaselineSystem,
    "xvec": XvecFitSystem,
}

# def get_system(algorithm, preprocess_config, model_config, train_config, algorithm_config, log_dir, result_dir):
    # return SYSTEM[algorithm](preprocess_config, model_config, train_config, algorithm_config, log_dir, result_dir)
def get_system(algorithm):
    return SYSTEM[algorithm]
