import pytorch_lightning as pl

from torch.utils.data import DataLoader, ConcatDataset

from .baseline_datamodule import BaselineDataModule
from .utils import few_shot_task_dataset


class MetaDataModule(BaselineDataModule):
    def __init__(self, preprocess_config, train_config, algorithm_config, log_dir, result_dir):
        super().__init__(preprocess_config, train_config, algorithm_config, log_dir, result_dir)


    def setup(self, stage=None):
        super(BaselineDataModule, self).setup(stage)
        # pl.seed_everything(43, True)

        if stage in (None, 'fit', 'validate'):
            self._train_setup()
            self._validation_setup()

        if stage in (None, 'test', 'predict'):
            self._test_setup()


    def _train_setup(self):
        epoch_length = self.meta_batch_size * self.val_step
        self.train_dataset = ConcatDataset(self.train_datasets)

        self.train_task_dataset = few_shot_task_dataset(
            self.train_dataset, self.train_ways, self.train_shots, self.train_queries,
            n_tasks_per_label=-1, epoch_length=epoch_length, type=self.meta_type
        )


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
