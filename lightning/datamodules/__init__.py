from .base_datamodule import BaseDataModule
from .baseline_datamodule import BaselineDataModule
from .meta_datamodule import MetaDataModule
from .xvec_datamodule import XvecDataModule


DATA_MODULE = {
    "base": BaseDataModule,
    "meta": MetaDataModule,
    "imaml": MetaDataModule,
    "baseline": BaselineDataModule,
    "xvec": XvecDataModule,
}

def get_datamodule(algorithm):
    return DATA_MODULE[algorithm]
