import torch
import pytorch_lightning as pl
from collections import defaultdict
from typing import Literal

from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.utils.data.dataset import ConcatDataset

from lightning.dataset import MonolingualTTSDataset
from lightning.collate import get_single_collate, reprocess
from lightning.utils import EpisodicInfiniteWrapper


class LearnablePruneDataModule(pl.LightningDataModule):
    """Datamodule for Xvec.

    Attributes:
        num_classes: Number of target labels.
    """

    def __init__(self,
                 preprocess_config: dict,
                 train_config: dict,
                 steps_per_epoch: int,
                 *args,
                 target: Literal["speaker", "region", "accent"] = "speaker",
                 m: int = 10,
                 k: int = 5,
                 q: int = 5,
                 key: str = None,
                 **kwargs):
        """Initialize PruneAccentDataModule.

        Args:
            preprocess_config:
                A dict at least containing paths required by MonolingualDataset.
            train_config:
                A dict at least containing batch_size.
            target:
                speaker/region/accent.
        """
        super().__init__()
        self.save_hyperparameters()
        self.preprocess_config = preprocess_config
        self.train_config = train_config

        self.target = target
        if self.target != "speaker":
            raise ValueError
        # self.dataset = MonolingualTTSDataset(
        #     preprocess_config["subsets"]["train"], self.preprocess_config,
        #     self.train_config)
        self.dataset = MonolingualTTSDataset("total", self.preprocess_config,
                                             self.train_config)
        self.num_classes = len(getattr(self.dataset, f"{self.target}_map"))
        self.steps_per_epoch = steps_per_epoch

        self.map_ids = defaultdict(list)
        for idx, tgt in enumerate(getattr(self.dataset, self.target)):
            if self.dataset.accent[idx] is None:
                continue
            self.map_ids[tgt].append(idx)

        # Few-shot settings: n-ways-k-shots-q-queries
        self.m = m  # data for training mask
        self.k = k
        self.q = q
        self.key = key
        keys = sorted(self.map_ids.keys())
        for _key in keys:
            if len(self.map_ids[_key]) < self.m + self.k + self.q:
                del self.map_ids[_key]

        if self.key is None:
            # generator = torch.Generator()
            # # generator.manual_seed(42)
            # generator.seed()
            # for i in torch.randperm(len(self.map_ids), generator=generator):
            for i in torch.randperm(len(self.map_ids)):
                self.key = sorted(self.map_ids.keys())[i]
        elif self.key not in self.map_ids:
            raise ValueError

    def setup(self, stage=None):
        if stage in (None, 'fit', 'validate', 'test'):
            key, train_mask_ids, train_sup_ids, train_qry_ids, val_ids = self._sample()
            print("Speaker", key)
            print("m-mask", list(train_mask_ids))
            print("k-shot", list(train_sup_ids))
            print("q-query", list(train_qry_ids))
            self.train_mask = reprocess(self.dataset, train_mask_ids) # dict of batch
            self.train_sup = reprocess(self.dataset, train_sup_ids) # dict of batch
            self.train_qry = reprocess(self.dataset, train_qry_ids) # dict of batch
            self.val_dataset = Subset(self.dataset, val_ids)

        if stage in (None, 'test'):
            if self.target == "speaker":
                other_spks = [pID for pID in self.map_ids
                              if ((self.dataset.speaker_accent_map[pID]
                                   == self.dataset.speaker_accent_map[key])
                                  and pID != key)]
                test_ids = sum([self.map_ids[pID] for pID in other_spks], [])
                if len(test_ids) > 0:
                    self.test_dataset = Subset(self.dataset, test_ids)

    def train_dataloader(self):
        """Training dataloader, not modified for multiple dataloaders."""
        self.train_dataset = [{
            "mask": self.train_mask,
            "sup": self.train_sup,
            "qry": self.train_qry,
        }]
        self.train_dataset = EpisodicInfiniteWrapper(
            self.train_dataset, self.steps_per_epoch)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            num_workers=4,
            collate_fn=lambda batch: batch,
        )
        return self.train_loader

    def val_dataloader(self):
        batch_size = self.train_config["optimizer"]["batch_size"]
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size//torch.cuda.device_count(),
            shuffle=False,
            drop_last=False,
            num_workers=4,
            collate_fn=get_single_collate(False),
        )
        return self.val_loader

    def test_dataloader(self):
        batch_size = self.train_config["optimizer"]["batch_size"]
        self.test_dataset = ConcatDataset(self.test_datasets)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size//torch.cuda.device_count(),
            shuffle=False,
            drop_last=False,
            num_workers=4,
            collate_fn=get_single_collate(False),
        )
        return self.test_loader

    def _sample(self):
        generator = torch.Generator()
        # generator.manual_seed(42)
        generator.seed()

        key = self.key
        split_len = [self.m, self.k, self.q,
                     len(self.map_ids[key]) - self.m - self.k - self.q]
        train_mask_ids, train_sup_ids, train_qry_ids, val_ids = random_split(
            self.map_ids[key], split_len, generator=generator
        )
        return key, train_mask_ids, train_sup_ids, train_qry_ids, val_ids


