from .base_datamodule import BaseDataModule
from .baseline_datamodule import BaselineDataModule
from .baselinev2_datamodule import BaselineV2DataModule
from .meta_datamodule import MetaDataModule


DATA_MODULE = {
    "base": BaseDataModule,
    "meta": MetaDataModule,
    "imaml": MetaDataModule,
    "asr-codebook": MetaDataModule,
    "asr-baseline": BaselineV2DataModule,
    "asr-center": BaselineV2DataModule,
    "asr-center-ref": MetaDataModule,
    "dual-meta": MetaDataModule,
    "dual-tune": BaselineDataModule,
    "meta-tune": BaselineDataModule,
    "baseline": BaselineDataModule,
}

def get_datamodule(algorithm):
    return DATA_MODULE[algorithm]
