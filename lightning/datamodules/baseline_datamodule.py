import torch

from torch.utils.data import DataLoader, ConcatDataset

from lightning.collate import get_single_collate
from lightning.utils import seed_all, EpisodicInfiniteWrapper

from .base_datamodule import BaseDataModule
from .utils import few_shot_task_dataset, prefetch_tasks, load_descriptions, get_SQids2Tid


class BaselineDataModule(BaseDataModule):
    def __init__(self, preprocess_config, train_config, algorithm_config, stage):
        super().__init__(preprocess_config, train_config, algorithm_config, stage)

        self.meta_type = self.algorithm_config["adapt"]["type"]

        self.train_ways = self.algorithm_config["adapt"]["train"]["ways"]
        self.train_shots = self.algorithm_config["adapt"]["train"]["shots"]
        self.train_queries = self.algorithm_config["adapt"]["train"]["queries"]

        self.test_ways = self.algorithm_config["adapt"]["test"]["ways"]
        self.test_shots = self.algorithm_config["adapt"]["test"]["shots"]
        self.test_queries = self.algorithm_config["adapt"]["test"]["queries"]

        self.meta_batch_size = self.algorithm_config["adapt"]["train"]["meta_batch_size"]
        self.log_step = self.train_config["step"]["log_step"]


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
            self.train_dataset = EpisodicInfiniteWrapper(self.train_dataset,
                                                         self.log_step*self.batch_size)


    def _validation_setup(self):
        self.val_task_datasets = []
        if not hasattr(self, "val_SQids2Tid"):
            self.val_SQids2Tid = {}
        if not hasattr(self, "val_task_descriptions"):
            self.val_task_descriptions = {}

        for i in range(len(self.val_datasets)):
            val_task_dataset = few_shot_task_dataset(
                ConcatDataset([self.val_datasets[i]]),
                self.test_ways, self.test_shots, self.test_queries,
                n_tasks_per_label=8, type=self.meta_type
            )
            self.val_task_datasets.append(val_task_dataset)

            dset_name = self.val_datasets[i].dset
            if dset_name in self.val_task_descriptions:
                descriptions = self.val_task_descriptions[dset_name]
                load_descriptions(val_task_dataset,
                                  loaded_descriptions=descriptions)
            else:
                with seed_all(43):
                    SQids, SQids2Tid = get_SQids2Tid(val_task_dataset,
                                                     tag=f"{dset_name}/val")
                self.val_SQids2Tid.update(SQids2Tid)
                descriptions = []
                for ds in val_task_dataset.datasets:
                    data_descriptions = {}
                    for i in ds.sampled_descriptions:
                        data_descriptions[i] = [desc.index for desc in
                                                ds.sampled_descriptions[i]]
                    descriptions.append(data_descriptions)
                self.val_task_descriptions[dset_name] = descriptions


    def state_dict(self):
        return {
            "val_SQids2Tid": self.val_SQids2Tid,
            "val_task_descriptions": self.val_task_descriptions,
        }


    def load_state_dict(self, state_dict):
        self.val_SQids2Tid = state_dict["val_SQids2Tid"]
        self.val_task_descriptions = state_dict["val_task_descriptions"]


    def _test_setup(self):
        # self.test_dataset = ConcatDataset(self.test_datasets)
        # self.test_task_dataset = few_shot_task_dataset(
        #     self.test_dataset, self.test_ways, self.test_shots, self.test_queries,
        #     n_tasks_per_label=16, type=self.meta_type
        # )
        # with seed_all(43):
        #     self.test_SQids2Tid = prefetch_tasks(self.test_task_dataset, 'test', self.result_dir)
        #TODO: remove self.result_dir
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
                num_workers=0,
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
