import os
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from typing import Any, Union, List, Tuple, Mapping
from pytorch_lightning import Trainer, LightningModule

from lightning.utils import loss2dict
from .base import BaseSaver


class ValidateSaver(BaseSaver):
    """
    """

    def __init__(self,
                 preprocess_config,
                 log_dir=None,
                 result_dir=None):
        super().__init__(preprocess_config, log_dir, result_dir)

    def on_validation_start(self,
                            trainer: Trainer,
                            pl_module: LightningModule):
        global_step = getattr(pl_module, 'test_global_step', pl_module.global_step)

        self.log_dir = os.path.join(trainer.log_dir, trainer.state.fn)
        self.result_dir = os.path.join(trainer.log_dir, trainer.state.fn)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        print("Log directory:", self.log_dir)
        print("Result directory:", self.result_dir)

        self.figure_dir = os.path.join(self.result_dir, f"figure/step-{global_step}")
        os.makedirs(self.figure_dir, exist_ok=True)

    def on_validation_batch_end(self,
                                trainer: Trainer,
                                pl_module: LightningModule,
                                outputs: Mapping,
                                batch: Any,
                                batch_idx: int,
                                dataloader_idx: int):
        global_step = getattr(pl_module, 'test_global_step', pl_module.global_step)
        test_adaptation_steps = max(pl_module.saving_steps) # total fine-tune steps

        SQids = '.'.join(['-'.join(batch[0][0][0]["ids"]),
                          '-'.join(batch[0][1][0]["ids"])])
        task_id = trainer.datamodule.val_SQids2Tid[SQids]

        self.plot_loss_fig(outputs,
                           test_adaptation_steps,
                           f"{trainer.state.stage}/{task_id}_losses",
                           save=True,
                           log=False,
                           logger=pl_module.logger,
                           global_step=global_step)

    def plot_loss_fig(self,
                      outputs: Mapping,
                      test_adaptation_steps: int,
                      figure_name: str,
                      save: bool = False,
                      log: bool = False,
                      logger: Any = None,
                      global_step: str = None):
        x = [ft_step for ft_step in range(test_adaptation_steps+1)
             if f"step_{ft_step}" in outputs]
        y = {
            "Total Loss": [],
            "Mel Loss": [],
            "Mel-Postnet Loss": [],
            "Pitch Loss": [],
            "Energy Loss": [],
            "Duration Loss": [],
        }
        for ft_step in x:
            loss_dict = loss2dict(outputs[f"step_{ft_step}"]["recon"]["losses"])
            for k in y:
                y[k].append(loss_dict[k])

        fig, axs = plt.subplots(nrows=2, ncols=3, constrained_layout=True)
        for ax, k in zip(axs.flat, y):
            ax.set_title(k)
            ax.plot(x, y[k], 'o', ls='-', ms=4)

        if log:
            self.log_figure(logger, figure_name, fig, global_step)

        if save:
            figure_path = os.path.join(self.figure_dir, f"{figure_name}.png")
            os.makedirs(os.path.dirname(figure_path), exist_ok=True)
            plt.savefig(figure_path)
        plt.close()
