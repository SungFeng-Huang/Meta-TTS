import pytorch_lightning as pl

from torch.utils.data import DataLoader

from dataset import Dataset
from lightning.collate import get_single_collate


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, preprocess_config, train_config, algorithm_config, log_dir, result_dir):
        super().__init__()
        self.preprocess_config = preprocess_config
        self.train_config = train_config
        self.algorithm_config = algorithm_config

        self.log_dir = log_dir
        self.result_dir = result_dir

    def setup(self, stage=None):
        refer_wav = self.algorithm_config["adapt"]["speaker_emb"] == "encoder"
        if stage in (None, 'fit', 'validate'):
            self.train_dataset = Dataset(
                f"{self.preprocess_config['subsets']['train']}.txt",
                self.preprocess_config, self.train_config, sort=True, drop_last=True, refer_wav=refer_wav
            )
            self.val_dataset = Dataset(
                f"{self.preprocess_config['subsets']['val']}.txt",
                self.preprocess_config, self.train_config, sort=False, drop_last=False, refer_wav=refer_wav
            )
        if stage in (None, 'test', 'predict'):
            self.test_dataset = Dataset(
                f"{self.preprocess_config['subsets']['test']}.txt",
                self.preprocess_config, self.train_config, sort=False, drop_last=False, refer_wav=refer_wav
            )

    def train_dataloader(self):
        """Training dataloader"""
        batch_size = self.train_config["optimizer"]["batch_size"]
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=8, collate_fn=get_single_collate(False),
        )
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        batch_size = self.train_config["optimizer"]["batch_size"]
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=8, collate_fn=get_single_collate(False),
        )
        return self.val_loader
