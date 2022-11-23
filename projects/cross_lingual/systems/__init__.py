from typing import Type

from . system import System
from . import language
from . import phoneme_recognition
from . import t2u


SYSTEM = {
    # "fscl": language.TransEmbSystem,
    # "fscl-tune": language.TransEmbTuneSystem,
    # "semi-fscl": language.SemiTransEmbSystem,
    # "semi-fscl-tune": language.SemiTransEmbTuneSystem,
    "baseline": language.BaselineSystem,
    # "baseline-tune": language.BaselineTuneSystem,

    "pr-ssl-linear-tune": phoneme_recognition.SSLLinearSystem,
    "pr-ssl-baseline": phoneme_recognition.SSLBaselineSystem,
    "pr-ssl-cluster": phoneme_recognition.SSLClusterSystem,
    # "pr-ssl-codebook-cluster": phoneme_recognition.SSLCodebookClusterSystem,
    "pr-ssl-baseline-tune": phoneme_recognition.SSLBaselineTuneSystem,
    "pr-ssl-cluster-tune": phoneme_recognition.SSLClusterTuneSystem,
    # "pr-fscl": phoneme_recognition.TransHeadSystem,
    # "pr-fscl-tune": phoneme_recognition.TransHeadTuneSystem,
    "pr-ssl-protonet": phoneme_recognition.SSLProtoNetSystem,

    "tacot2u": t2u.TacoT2USystem,
}

def get_system(algorithm) -> Type[System]:
    return SYSTEM[algorithm]
