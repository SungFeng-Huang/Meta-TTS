from torch.utils.data.dataset import ConcatDataset
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from dataset import MonolingualTTSDataset as Dataset
from dataset import TextDataset2
from lightning.collate import get_single_collate, LanguageTaskCollate


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
                ) for preprocess_config in self.preprocess_configs if 'train' in preprocess_config['subsets']
            ]
            self.val_datasets = [
                Dataset(
                    f"{preprocess_config['subsets']['val']}.txt",
                    preprocess_config, self.train_config, sort=False, drop_last=False, spk_refer_wav=spk_refer_wav
                ) for preprocess_config in self.preprocess_configs if 'val' in preprocess_config['subsets']
            ]
            self.train_dataset = ConcatDataset(self.train_datasets)
            self.val_dataset = ConcatDataset(self.val_datasets)

        if stage in (None, 'test', 'predict'):
            self.test_datasets = [
                TextDataset2(f"{preprocess_config['path']['preprocessed_path']}/{preprocess_config['subsets']['test']}.txt", preprocess_config) 
                for preprocess_config in self.preprocess_configs if 'test' in preprocess_config['subsets']
            ]
            self.test_dataset = ConcatDataset(self.test_datasets)
