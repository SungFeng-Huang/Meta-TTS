from torch.utils.data.dataset import ConcatDataset
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from dataset import MonolingualTTSDataset as Dataset
from lightning.collate import get_single_collate


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, preprocess_configs, train_config, algorithm_config, log_dir, result_dir):
        super().__init__()
        self.preprocess_configs = preprocess_configs
        self.train_config = train_config
        self.algorithm_config = algorithm_config

        self.log_dir = log_dir
        self.result_dir = result_dir


    def setup(self, stage=None):
        spk_refer_wav = (self.algorithm_config["adapt"]["speaker_emb"]
                     in ["dvec", "encoder", "scratch_encoder"])

        if stage in (None, 'fit', 'validate'):
            self.train_datasets = [
                Dataset(
                    f"{preprocess_config['subsets']['train']}.txt",
                    preprocess_config, self.train_config, sort=True, drop_last=True, spk_refer_wav=spk_refer_wav
                ) for preprocess_config in self.preprocess_configs
            ]
            self.val_datasets = [
                Dataset(
                    f"{preprocess_config['subsets']['val']}.txt",
                    preprocess_config, self.train_config, sort=False, drop_last=False, spk_refer_wav=spk_refer_wav
                ) for preprocess_config in self.preprocess_configs
            ]

        if stage in (None, 'test', 'predict'):
            self.test_datasets = [
                Dataset(
                    f"{preprocess_config['subsets']['test']}.txt",
                    preprocess_config, self.train_config, sort=False, drop_last=False, spk_refer_wav=spk_refer_wav
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
