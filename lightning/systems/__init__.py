# from .baseline import BaselineSystem
# from .meta import MetaSystem
from . import TTS
from . import ASR
from .imaml import IMAMLSystem
from . import Dual

SYSTEM = {
    "meta": TTS.maml.MetaSystem,
    "imaml": IMAMLSystem,
    "baseline": TTS.baseline.BaselineSystem,
    "asr-codebook": ASR.codebook.CodebookSystem,
    "asr-baseline": ASR.center.CenterSystem,
    "asr-center": ASR.center.CenterSystem,
    "asr-center-ref": ASR.center_ref.CenterRefSystem,
    "dual-meta": Dual.dual.DualMetaSystem,
}

# def get_system(algorithm, preprocess_config, model_config, train_config, algorithm_config, log_dir, result_dir):
    # return SYSTEM[algorithm](preprocess_config, model_config, train_config, algorithm_config, log_dir, result_dir)
def get_system(algorithm):
    return SYSTEM[algorithm]
