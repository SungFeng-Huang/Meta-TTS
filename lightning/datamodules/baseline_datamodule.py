import os
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader, ConcatDataset

from lightning.collate import get_single_collate
from lightning.utils import seed_all, EpisodicInfiniteWrapper

from .base_datamodule import BaseDataModule
from .utils import few_shot_task_dataset, prefetch_tasks


class BaselineDataModule(BaseDataModule):
    def __init__(self, preprocess_config, train_config, algorithm_config,
                 log_dir, result_dir, stage):
        super().__init__(preprocess_config, train_config, algorithm_config,
                         log_dir, result_dir, stage)

        self.meta_type = self.algorithm_config["adapt"]["type"]

        self.train_ways = self.algorithm_config["adapt"]["train"]["ways"]
        self.train_shots = self.algorithm_config["adapt"]["train"]["shots"]
        self.train_queries = self.algorithm_config["adapt"]["train"]["queries"]

        self.test_ways = self.algorithm_config["adapt"]["test"]["ways"]
        self.test_shots = self.algorithm_config["adapt"]["test"]["shots"]
        self.test_queries = self.algorithm_config["adapt"]["test"]["queries"]

        self.meta_batch_size = self.algorithm_config["adapt"]["train"]["meta_batch_size"]
        self.val_step = self.train_config["step"]["val_step"]


    def setup(self, stage=None):
        super().setup(stage)
        # pl.seed_everything(43, True)

        if stage in (None, 'fit'):
            self._train_setup()

        if stage in (None, 'fit', 'validate'):
            self._validation_setup()

        if stage in (None, 'test', 'predict'):
            self._test_setup()


    def _train_setup(self):
        self.train_dataset = ConcatDataset(self.train_datasets)
        if not isinstance(self.train_dataset, EpisodicInfiniteWrapper):
            self.batch_size = self.train_ways * (self.train_shots + self.train_queries) * self.meta_batch_size
            self.train_dataset = EpisodicInfiniteWrapper(self.train_dataset, self.val_step*self.batch_size)


    def _validation_setup(self):
        self.val_SQids2Tid = {}
        self.val_task_datasets = []

        for i in range(len(self.val_datasets)):
            dset_name = self.val_datasets[i].dset
            log_dir = os.path.join(self.log_dir, "metadata", dset_name)
            SQids_filename = os.path.join(log_dir, f"val_SQids.json")
            desc_filename = os.path.join(log_dir, f'val_descriptions.json')

            val_task_dataset = few_shot_task_dataset(
                ConcatDataset([self.val_datasets[i]]),
                self.test_ways, self.test_shots, self.test_queries,
                n_tasks_per_label=8, type=self.meta_type
            )
            self.val_task_datasets.append(val_task_dataset)

            with seed_all(43):
                val_SQids2Tid = prefetch_tasks(
                    val_task_dataset,
                    tag=f"{dset_name}/val",
                    SQids_filename=SQids_filename,
                    desc_filename=desc_filename,
                )
            self.val_SQids2Tid.update(val_SQids2Tid)


    def _test_setup(self):
        # self.test_dataset = ConcatDataset(self.test_datasets)
        # self.test_task_dataset = few_shot_task_dataset(
        #     self.test_dataset, self.test_ways, self.test_shots, self.test_queries,
        #     n_tasks_per_label=16, type=self.meta_type
        # )
        # with seed_all(43):
        #     self.test_SQids2Tid = prefetch_tasks(self.test_task_dataset, 'test', self.result_dir)
        self.test_task_datasets = [
            few_shot_task_dataset(
                ConcatDataset([test_dataset]),
                self.test_ways, self.test_shots, self.test_queries,
                n_tasks_per_label=16, type=self.meta_type
            ) for test_dataset in self.test_datasets
        ]
        with seed_all(43):
            self.test_SQids2Tid = prefetch_tasks(
                ConcatDataset(self.test_task_datasets), 'test', self.result_dir
            )


    def train_dataloader(self):
        """Training dataloader"""
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size//torch.cuda.device_count(),
            shuffle=True,
            drop_last=True,
            num_workers=4,
            collate_fn=get_single_collate(False),
        )
        return self.train_loader


    def val_dataloader(self):
        """Validation dataloader"""
        self.val_loaders = [
            DataLoader(
                val_task_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=4,
                collate_fn=lambda batch: batch,
            ) for val_task_dataset in self.val_task_datasets
        ]
        return self.val_loaders


    def test_dataloader(self):
        """Test dataloader"""
        self.test_loaders = [
            DataLoader(
                test_task_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=4,
                collate_fn=lambda batch: batch,
            ) for test_task_dataset in self.test_task_datasets
        ]
        return self.test_loaders
