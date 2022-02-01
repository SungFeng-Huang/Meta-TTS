# from .baseline import BaselineSystem
# from .meta import MetaSystem
from . import TTS
from . import ASR
from .imaml import IMAMLSystem
from .dual import DualSystem

SYSTEM = {
    "meta": TTS.maml.MetaSystem,
    "imaml": IMAMLSystem,
    "baseline": TTS.baseline.BaselineSystem,
    "asr-codebook": ASR.codebook.CodebookSystem,
    "asr-baseline": ASR.baseline.BaselineSystem,
}

# def get_system(algorithm, preprocess_config, model_config, train_config, algorithm_config, log_dir, result_dir):
    # return SYSTEM[algorithm](preprocess_config, model_config, train_config, algorithm_config, log_dir, result_dir)
def get_system(algorithm):
    return SYSTEM[algorithm]
