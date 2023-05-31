import sys
import os
import torch
import pytest

from torch.testing import assert_allclose, assert_close
from collections import OrderedDict

from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer

from projects.prune.systems.post_maml import PrunedMetaSystem


algorithm_config = {
    "adapt": {
        "task": {
            "lr": 0.001,
            "weight_decay": 0.01,
        }
    }
}

pruned_ckpt_path = "output/libri_masked_pruned.pth"

def test_pruned_meta_system():
    system = PrunedMetaSystem(
        pruned_ckpt_path=pruned_ckpt_path,
        algorithm_config=algorithm_config,
    )