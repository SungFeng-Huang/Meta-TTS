from typing import Type

from . system import System
from .FastSpeech2 import BaselineSystem
from .TransEmb import TransEmbSystem
from .FastSpeech2Tune import BaselineTuneSystem
from .TransEmbTune import TransEmbTuneSystem


SYSTEM = {
    "baseline": BaselineSystem,
    "fscl": TransEmbSystem,

    "baseline-tune": BaselineTuneSystem,
    "fscl-tune": TransEmbTuneSystem,
}


def get_system(algorithm) -> Type[System]:
    return SYSTEM[algorithm]
