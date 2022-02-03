import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader, ConcatDataset

from text.define import LANG_ID2SYMBOLS
from lightning.collate import LanguageTaskCollate, get_single_collate
from lightning.utils import seed_all, EpisodicInfiniteWrapper

from .base_datamodule import BaseDataModule
from .utils import few_shot_task_dataset, prefetch_tasks


class BaselineDataModule(BaseDataModule):
    def __init__(self, preprocess_config, train_config, algorithm_config, log_dir, result_dir):
        super().__init__(preprocess_config, train_config, algorithm_config, log_dir, result_dir)
        self.meta_type = self.algorithm_config["adapt"]["type"]

        self.train_ways = self.algorithm_config["adapt"]["train"]["ways"]
        self.train_shots = self.algorithm_config["adapt"]["train"]["shots"]
        self.train_queries = self.algorithm_config["adapt"]["train"]["queries"]

        self.test_ways = self.algorithm_config["adapt"]["test"]["ways"]
        self.test_shots = self.algorithm_config["adapt"]["test"]["shots"]
        self.test_queries = self.algorithm_config["adapt"]["test"]["queries"]

        self.meta_batch_size = self.algorithm_config["adapt"]["train"]["meta_batch_size"]
        self.val_step = self.train_config["step"]["val_step"]

        if self.meta_type == "lang":
            self.collate = LanguageTaskCollate({
                "lang_id2symbols": LANG_ID2SYMBOLS,
                "representation_dim": 1024,
            })
        elif self.meta_type == "spk":
            pass
        else:
            raise NotImplementedError


    def setup(self, stage=None):
        super().setup(stage)
        # pl.seed_everything(43, True)

        if stage in (None, 'fit', 'validate'):
            self._train_setup()
            self._validation_setup()

        if stage in (None, 'test', 'predict'):
            self._test_setup()


    def _train_setup(self):
        # self.train_dataset = ConcatDataset(self.train_datasets)
        if not isinstance(self.train_dataset, EpisodicInfiniteWrapper):
            self.batch_size = self.train_ways * (self.train_shots + self.train_queries) * self.meta_batch_size
            self.train_dataset = EpisodicInfiniteWrapper(self.train_dataset, self.val_step*self.batch_size)


    def _validation_setup(self):
        # self.val_dataset = ConcatDataset(self.val_datasets)
        # self.val_task_dataset = few_shot_task_dataset(
        #     self.val_dataset, self.test_ways, self.test_shots, self.test_queries,
        #     n_tasks_per_label=8, type=self.meta_type
        # )
        # with seed_all(43):
        #     self.val_SQids2Tid = prefetch_tasks(self.val_task_dataset, 'val', self.log_dir)
        pass


    def _test_setup(self):
        # self.test_dataset = ConcatDataset(self.test_datasets)
        self.test_task_dataset = few_shot_task_dataset(
            self.test_dataset, self.test_ways, self.test_shots, self.test_queries,
            n_tasks_per_label=1, type=self.meta_type, re_id=True
        )
        with seed_all(43):
            self.test_SQids2Tid = prefetch_tasks(self.test_task_dataset, 'test', self.result_dir)


    def train_dataloader(self):
        """Training dataloader, not modified for multiple dataloaders."""
        batch_size = self.train_config["optimizer"]["batch_size"]
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size//torch.cuda.device_count(),
            shuffle=True,
            drop_last=True,
            num_workers=4,
            collate_fn=self.collate.get_collate(False),
        )
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader, not modified for multiple dataloaders."""
        batch_size = self.train_config["optimizer"]["batch_size"]
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size//torch.cuda.device_count(),
            shuffle=False,
            drop_last=False,
            num_workers=4,
            collate_fn=self.collate.get_collate(False),
        )
        return self.val_loader


    def test_dataloader(self):
        """Test dataloader"""
        self.test_loader = DataLoader(
            self.test_task_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: batch,
        )
        return self.test_loader
