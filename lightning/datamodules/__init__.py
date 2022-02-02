from .base_datamodule import BaseDataModule
from .baseline_datamodule import BaselineDataModule
from .meta_datamodule import MetaDataModule


DATA_MODULE = {
    "base": BaseDataModule,
    "meta": MetaDataModule,
    "imaml": MetaDataModule,
    "asr-codebook": MetaDataModule,
    "asr-baseline": BaselineDataModule,
    "dual-meta": MetaDataModule,
    "baseline": BaselineDataModule,
}

def get_datamodule(algorithm):
    return DATA_MODULE[algorithm]
