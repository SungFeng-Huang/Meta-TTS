from tqdm.auto import tqdm

import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.progress import ProgressBarBase


class GlobalProgressBar(Callback):
    """Global progress bar.
    TODO: add progress bar for training, validation and testing loop.
    """

    def __init__(self, global_progress: bool = True, leave_global_progress: bool = True, process_position=0):
        super().__init__()

        self.global_progress = global_progress
        self.global_desc = "Steps: {steps}/{max_steps}"
        self.leave_global_progress = leave_global_progress
        self.global_pb = None
        self.process_position = process_position

    def on_train_start(self, trainer, pl_module):
        desc = self.global_desc.format(steps=pl_module.global_step + 1, max_steps=trainer.max_steps)

        self.global_pb = tqdm(
            desc=desc,
            dynamic_ncols=True,
            total=trainer.max_steps,
            initial=pl_module.global_step,
            leave=self.leave_global_progress,
            disable=not self.global_progress,
            position=self.process_position,
            file=sys.stdout,
        )

    def on_train_end(self, trainer, pl_module):
        self.global_pb.close()
        self.global_pb = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):

        # Set description
        desc = self.global_desc.format(steps=pl_module.global_step + 1, max_steps=trainer.max_steps)
        self.global_pb.set_description(desc)

        # Update progress
        if (pl_module.global_step+1) % trainer.accumulate_grad_batches == 0:
            self.global_pb.update(1)

