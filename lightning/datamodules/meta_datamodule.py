import pytorch_lightning as pl

from torch.utils.data import DataLoader

from lightning.utils import seed_all

from .base_datamodule import BaseDataModule
from .utils import few_shot_task_dataset, prefetch_tasks


class MetaDataModule(BaseDataModule):
    def __init__(self, preprocess_configs, train_config, algorithm_config, log_dir, result_dir):
        super().__init__(preprocess_configs, train_config,
                         algorithm_config, log_dir, result_dir)
        self.meta_type = self.algorithm_config["meta_type"]
        self.train_ways = self.algorithm_config["adapt"]["ways"]
        self.train_shots = self.algorithm_config["adapt"]["shots"]
        self.train_queries = self.algorithm_config["adapt"]["queries"]

        self.test_ways = self.algorithm_config["adapt"]["ways"]
        self.test_shots = self.algorithm_config["adapt"]["shots"]
        self.test_queries = self.algorithm_config["adapt"]["test"]["queries"]

        self.meta_batch_size = self.algorithm_config["adapt"]["meta_batch_size"]
        self.val_step = self.train_config["step"]["val_step"]

    def setup(self, stage=None):
        super().setup(stage)
        # pl.seed_everything(43, True)

        if stage in (None, 'fit', 'validate'):
            # Train set
            epoch_length = self.meta_batch_size * self.val_step
            self.train_task_dataset = few_shot_task_dataset(
                self.train_datasets, self.train_ways, self.train_shots, self.train_queries,
                task_per_speaker=-1, epoch_length=epoch_length, type=self.meta_type
            )

            # Validation set
            self.val_task_dataset = few_shot_task_dataset(
                self.val_datasets, self.test_ways, self.test_shots, self.test_queries, task_per_speaker=8, type=self.meta_type
            )
            with seed_all(43):
                self.val_SQids2Tid = prefetch_tasks(
                    self.val_task_dataset, 'val', self.log_dir)

        if stage in (None, 'test', 'predict'):
            # Test set
            self.test_task_dataset = few_shot_task_dataset(
                self.test_datasets, self.test_ways, self.test_shots, self.test_queries, task_per_speaker=16, type=self.meta_type
            )
            with seed_all(43):
                self.test_SQids2Tid = prefetch_tasks(
                    self.test_task_dataset, 'test', self.result_dir)

    def train_dataloader(self):
        """Training dataloader"""
        self.train_loader = DataLoader(
            self.train_task_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            collate_fn=lambda batch: batch,
        )
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        self.val_loader = DataLoader(
            self.val_task_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            collate_fn=lambda batch: batch,
        )
        return self.val_loader

    def test_dataloader(self):
        """Test dataloader"""
        self.test_loader = DataLoader(
            self.test_task_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            collate_fn=lambda batch: batch,
        )
        return self.test_loader
