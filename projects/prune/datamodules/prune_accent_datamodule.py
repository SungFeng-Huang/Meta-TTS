import yaml
import torch
import pytorch_lightning as pl
from collections import defaultdict
from typing import Literal, Dict, Any, Union, List

from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.utils.data.dataset import ConcatDataset

from lightning.dataset import MonolingualTTSDataset
from lightning.collate import get_single_collate, reprocess
from lightning.utils import EpisodicInfiniteWrapper


def load_yaml(yaml_in: str) -> Dict[str, Any]:
    return yaml.load(open(yaml_in), Loader=yaml.FullLoader)


class PruneAccentDataModule(pl.LightningDataModule):
    """Datamodule for Xvec.

    Attributes:
        num_classes: Number of target labels.
    """

    def __init__(self,
                 preprocess_config: Union[dict, str],
                 train_config: Union[dict, str, List[str]],
                 steps_per_epoch: int,
                 *args,
                 target: Literal["speaker", "region", "accent"] = "speaker",
                 mask_steps_per_epoch: int = 0,
                 m: int = 0,
                 k: int = 5,
                 q: int = 5,
                 key: str = None,
                 batch_size: int = None,
                 libri_mask: bool = False,
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

        if isinstance(preprocess_config, str):
            preprocess_config = load_yaml(preprocess_config)
        if isinstance(train_config, str):
            train_config = load_yaml(train_config)
        elif isinstance(train_config, List):
            _configs = [load_yaml(_config) for _config in train_config]
            train_config = _configs[0]
            for _config in _configs[1:]:
                train_config.update(_config)

        if batch_size is not None:
            train_config['optimizer']['batch_size'] = batch_size

        self.preprocess_config = preprocess_config
        self.train_config = train_config
        # self.save_hyperparameters()

        self.target = target
        if self.target != "speaker":
            raise ValueError
        self.dataset = MonolingualTTSDataset(
            "total", # preprocess_config["subsets"]["train"]
            self.preprocess_config, self.train_config)
        self.num_classes = len(getattr(self.dataset, f"{self.target}_map"))
        self.steps_per_epoch = steps_per_epoch
        self.mask_steps_per_epoch = mask_steps_per_epoch

        self.map_ids = defaultdict(list)
        for idx, tgt in enumerate(getattr(self.dataset, self.target)):
            if self.dataset.accent[idx] is None:
                continue
            self.map_ids[tgt].append(idx)

        self.libri_mask = libri_mask
        if libri_mask:
            self.libri_dataset = MonolingualTTSDataset(
                "train", self.preprocess_config, self.train_config)

        # Few-shot settings: n-ways-k-shots-q-queries
        self.m = m  # data for training mask
        self.k = k
        self.q = q
        keys = sorted(self.map_ids.keys())
        for _key in keys:
            if len(self.map_ids[_key]) < self.m + self.k + self.q:
                del self.map_ids[_key]

        if key is None:
            # generator = torch.Generator()
            # # generator.manual_seed(42)
            # generator.seed()
            # for i in torch.randperm(len(self.map_ids), generator=generator):
            for i in torch.randperm(len(self.map_ids)):
                self.key = sorted(self.map_ids.keys())[i]
        elif key not in self.map_ids:
            raise ValueError
        else:
            self.key = key

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
        if not self.libri_mask:
            if not hasattr(self, "pipeline_stage"):
                self.train_loader = self.fixed_mask_steps_dataloader()
            elif self.pipeline_stage == "mask":
                self.train_loader = self.mask_stage_dataloader()
            elif self.pipeline_stage == "ft":
                self.train_loader = self.ft_stage_dataloader()
            elif self.pipeline_stage == "joint":
                self.train_loader = self.joint_stage_dataloader()
        else:
            if self.pipeline_stage == "mask":
                self.train_loader = {
                    "mask": self.libri_mask_stage_dataloader(),
                    "ft": self.ft_stage_dataloader(),
                }
            elif self.pipeline_stage == "ft":
                self.train_loader = {
                    "ft": self.ft_stage_dataloader(),
                }
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
        # generator = torch.Generator()
        # # generator.manual_seed(42)
        # generator.seed()

        key = self.key
        split_len = [self.m, self.k, self.q,
                     len(self.map_ids[key]) - self.m - self.k - self.q]
        train_mask_ids, train_sup_ids, train_qry_ids, val_ids = random_split(
            self.map_ids[key], split_len,
            # generator=generator,
        )
        if self.m == 0:
            train_mask_ids = train_sup_ids
        return key, train_mask_ids, train_sup_ids, train_qry_ids, val_ids

    def libri_mask_stage_dataloader(self):
        mask_dataset = EpisodicInfiniteWrapper(self.libri_dataset, self.steps_per_epoch)
        train_loader = DataLoader(
            mask_dataset,
            batch_size=self.train_config["optimizer"]["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=4,
            collate_fn=get_single_collate(False),
        ) 
        return train_loader

    def mask_stage_dataloader(self):
        mask_dataset = [
            {
                "mask": self.train_mask,
                "qry": self.train_qry,
            }
            for _ in range(self.steps_per_epoch)
        ]
        train_loader = DataLoader(
            mask_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            num_workers=4,
            collate_fn=lambda batch: batch,
        )
        return train_loader

    def ft_stage_dataloader(self):
        # ft_dataset for fine-tuning TTS parameters
        ft_dataset = [
            {
                "sup": self.train_sup,
                "qry": self.train_qry,
            }
            for _ in range(self.steps_per_epoch)
        ]
        train_loader = DataLoader(
            ft_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            num_workers=4,
            collate_fn=lambda batch: batch,
        )
        return train_loader

    def joint_stage_dataloader(self):
        joint_dataset = [{
            "mask": self.train_mask,
            "sup": self.train_sup,
            "qry": self.train_qry,
        }] * self.steps_per_epoch
        train_loader = DataLoader(
            joint_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            num_workers=4,
            collate_fn=lambda batch: batch,
        )
        return train_loader

    def fixed_mask_steps_dataloader(self):
        """ Theoretically won't happen for pipeline series."""
        train_dataset = [{
            "mask": self.train_mask,
            "qry": self.train_qry,
        }] * self.mask_steps_per_epoch + [{
            "sup": self.train_sup,
            "qry": self.train_qry,
        }] * (self.steps_per_epoch - self.mask_steps_per_epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            num_workers=4,
            collate_fn=lambda batch: batch,
        )
        return train_loader
