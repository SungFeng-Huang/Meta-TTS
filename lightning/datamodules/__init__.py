from .baseline_datamodule import BaselineDataModule
from .meta_datamodule import MetaDataModule


DATA_MODULE = {
    "meta": MetaDataModule,
    "baseline": BaselineDataModule,
}

def get_datamodule(algorithm):
    return DATA_MODULE[algorithm]
