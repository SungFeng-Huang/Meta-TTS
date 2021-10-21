from .base_datamodule import BaseDataModule
from .baseline_datamodule import BaselineDataModule
from .meta_datamodule import MetaDataModule
from .ml_meta_datamodule import MultilingualMetaDataModule


DATA_MODULE = {
    "base": BaseDataModule,
    "meta": MetaDataModule,
    "baseline": BaselineDataModule,
    "ml-meta": MultilingualMetaDataModule
}


def get_datamodule(algorithm):
    return DATA_MODULE[algorithm]
