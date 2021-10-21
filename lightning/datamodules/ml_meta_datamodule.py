from torch.utils.data.dataset import ConcatDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from lightning.utils import seed_all
from dataset import MultiLingualDataset
from .utils import multilingual_few_shot_task_dataset, prefetch_tasks, get_multilingual_id2lb


class MultilingualMetaDataModule(pl.LightningDataModule):
    def __init__(self, preprocess_configs, train_config, algorithm_config, log_dir, result_dir):
        super().__init__()
        self.preprocess_configs = preprocess_configs
        self.train_config = train_config
        self.algorithm_config = algorithm_config

        self.log_dir = log_dir
        self.result_dir = result_dir

        self.train_ways = self.algorithm_config["adapt"]["ways"]
        self.train_shots = self.algorithm_config["adapt"]["shots"]
        self.train_queries = self.algorithm_config["adapt"]["queries"]

        self.test_ways = self.algorithm_config["adapt"]["ways"]
        self.test_shots = self.algorithm_config["adapt"]["shots"]
        self.test_queries = self.algorithm_config["adapt"]["test"]["queries"]

        self.meta_batch_size = self.algorithm_config["adapt"]["meta_batch_size"]
        self.val_step = self.train_config["step"]["val_step"]

    def setup(self, stage=None):
        refer_wav = self.algorithm_config["adapt"]["speaker_emb"] in [
            "dvec", "encoder", "scratch_encoder"]
        if stage in (None, 'fit', 'validate'):
            train_datasets = [MultiLingualDataset(
                f"{preprocess_config['subsets']['train']}.txt",
                preprocess_config, self.train_config, sort=True, drop_last=True, refer_wav=refer_wav
            ) for preprocess_config in self.preprocess_configs]
            val_datasets = [MultiLingualDataset(
                f"{preprocess_config['subsets']['val']}.txt",
                preprocess_config, self.train_config, sort=False, drop_last=False, refer_wav=refer_wav
            ) for preprocess_config in self.preprocess_configs]

            # Train set
            epoch_length = self.meta_batch_size * self.val_step
            self.train_task_dataset = multilingual_few_shot_task_dataset(
                train_datasets, self.train_ways, self.train_shots, self.train_queries, task_per_label=-1, epoch_length=epoch_length
            )

            # Validation set
            self.val_task_dataset = multilingual_few_shot_task_dataset(
                val_datasets, self.test_ways, self.test_shots, self.test_queries, task_per_label=8,
            )
            # with seed_all(43):
            #     self.val_SQids2Tid = prefetch_tasks(
            #         self.val_task_dataset, 'val', self.log_dir)

        if stage in (None, 'test', 'predict'):
            test_datasets = [MultiLingualDataset(
                f"{preprocess_config['subsets']['test']}.txt",
                preprocess_config, self.train_config, sort=False, drop_last=False, refer_wav=refer_wav
            ) for preprocess_config in self.preprocess_configs]
            # Test set
            self.test_task_dataset = multilingual_few_shot_task_dataset(
                test_datasets, self.test_ways, self.test_shots, self.test_queries, task_per_label=16,
            )
            # with seed_all(43):
            #     self.test_SQids2Tid = prefetch_tasks(
            #         self.test_task_dataset, 'test', self.result_dir)

    def train_dataloader(self):
        """Training dataloader"""
        self.train_loader = DataLoader(
            self.train_task_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda batch: batch,
        )
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        self.val_loader = DataLoader(
            self.val_task_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: batch,
        )
        return self.val_loader

    def test_dataloader(self):
        """Test dataloader"""
        self.test_loader = DataLoader(
            self.test_task_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: batch,
        )
        return self.test_loader
