from typing import Literal, Any

import torch

from .xvec import XvecDataModule
from ..dataset import XvecOnlineDataset as Dataset


class XvecOnlineDataModule(XvecDataModule):

    dataset_cls = Dataset

    def __init__(self,
                 preprocess_config: dict,
                 train_config: dict,
                 *args,
                 dset = "total",
                 target: Literal["speaker", "region", "accent"] = "speaker"):
        """Initialize XvecDataModule.

        Args:
            preprocess_config:
                A dict at least containing paths required by XvecDataset.
            train_config:
                A dict at least containing batch_size.
            target:
                speaker/region/accent.
        """
        # change self.dataset_cls to onlinedataset
        super().__init__(preprocess_config, train_config, target=target)

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        super().on_before_batch_transfer(batch, dataloader_idx)

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        super().transfer_batch_to_device(batch, device, dataloader_idx)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        super().on_after_batch_transfer(batch, dataloader_idx)
