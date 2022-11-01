import torch

from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
import pytorch_lightning as pl

from lightning.dataset import MonolingualTTSDataset as Dataset
from lightning.collate import get_single_collate

import yaml
from typing import Literal, Dict, Any, Union, List


def load_yaml(yaml_in: str) -> Dict[str, Any]:
    return yaml.load(open(yaml_in), Loader=yaml.FullLoader)


class BaseDataModule(pl.LightningDataModule):
    def __init__(self,
                 preprocess_config: Union[dict, str, list],
                 train_config: Union[dict, str, list],
                 algorithm_config: Union[dict, str],
                 stage: str = "train"):
        super().__init__()

        if (isinstance(preprocess_config, str)
                or isinstance(preprocess_config, dict)):
            preprocess_config = [preprocess_config]
        _preprocess_configs = []
        for _config in preprocess_config:
            if isinstance(_config, str):
                _preprocess_configs.append(load_yaml(_config))
            elif isinstance(_config, dict):
                _preprocess_configs.append(_config)
        preprocess_configs = _preprocess_configs
        self.preprocess_configs = preprocess_configs

        if isinstance(train_config, str) or isinstance(train_config, dict):
            train_config = [train_config]
        _train_config = {}
        for _config in train_config:
            if isinstance(_config, str):
                _train_config.update(load_yaml(_config))
            elif isinstance(_config, dict):
                _train_config.update(_config)
        train_config = _train_config
        self.train_config = train_config

        if isinstance(algorithm_config, str):
            algorithm_config = load_yaml(algorithm_config)
        self.algorithm_config = algorithm_config

        # Discriminate train/transfer
        self.stage = stage


    def setup(self, stage=None):
        spk_refer_wav = (self.algorithm_config["adapt"]["speaker_emb"]
                         in ["dvec", "encoder", "scratch_encoder"])

        if stage in (None, 'fit'):
            self.train_datasets = [
                Dataset(
                    self.stage,
                    preprocess_config, self.train_config, spk_refer_wav=spk_refer_wav
                ) for preprocess_config in self.preprocess_configs
            ]

        if stage in (None, 'fit', 'validate'):
            self.val_datasets = []
            for preprocess_config in self.preprocess_configs:
                self.val_datasets += [
                    Dataset(
                        dset,
                        preprocess_config, self.train_config, spk_refer_wav=spk_refer_wav
                    ) for dset in preprocess_config["subsets"]["val"]
                ]

        if stage in (None, 'test', 'predict'):
            self.test_datasets = [
                Dataset(
                    "test",
                    preprocess_config, self.train_config, spk_refer_wav=spk_refer_wav
                ) for preprocess_config in self.preprocess_configs
            ]


    def train_dataloader(self):
        """Training dataloader, not modified for multiple dataloaders."""
        batch_size = self.train_config["optimizer"]["batch_size"]
        self.train_dataset = ConcatDataset(self.train_datasets)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size//torch.cuda.device_count(),
            shuffle=True,
            drop_last=True,
            num_workers=4,
            collate_fn=get_single_collate(False),
        )
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader, not modified for multiple dataloaders."""
        batch_size = self.train_config["optimizer"]["batch_size"]
        self.val_dataset = ConcatDataset(self.val_datasets)
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size//torch.cuda.device_count(),
            shuffle=False,
            drop_last=False,
            num_workers=4,
            collate_fn=get_single_collate(False),
        )
        return self.val_loader
