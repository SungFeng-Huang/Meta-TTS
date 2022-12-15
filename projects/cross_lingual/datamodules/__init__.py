from .FastSpeech2DataModule import FastSpeech2DataModule, FastSpeech2TuneDataModule
from .FSCLDataModule import FSCLDataModule, UnsupFSCLDataModule, SemiFSCLDataModule, SemiFSCLTuneDataModule


from . import language
from . import phoneme_recognition
from . import t2u


DATA_MODULE = {
    # "fscl": language.FSCLDataModule,
    # "fscl-tune": language.FastSpeech2DataModule,
    # "semi-fscl": language.SemiFSCLDataModule,
    # "semi-fscl-tune": language.SemiFSCLTuneDataModule,
    "baseline": language.FastSpeech2DataModule,
    # "baseline-tune": language.FastSpeech2TuneDataModule,

    "pr-ssl-linear-tune": phoneme_recognition.SSLPRDataModule,
    "pr-ssl-baseline": phoneme_recognition.SSLPRDataModule,
    "pr-ssl-cluster": phoneme_recognition.SSLPRDataModule,
    # "pr-ssl-codebook-cluster": phoneme_recognition.SSLPRDataModule,
    "pr-ssl-baseline-tune": phoneme_recognition.SSLPRDataModule,
    "pr-ssl-cluster-tune": phoneme_recognition.SSLPRDataModule,
    # "pr-fscl": phoneme_recognition.FSCLDataModule,
    # "pr-fscl-tune": phoneme_recognition.SSLPRDataModule,
    "pr-ssl-protonet": phoneme_recognition.FSCLDataModule,

    "tacot2u": t2u.T2UDataModule,
}

def get_datamodule(algorithm):
    return DATA_MODULE[algorithm]
