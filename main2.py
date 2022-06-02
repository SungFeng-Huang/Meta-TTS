import argparse
import os
import tempfile

import comet_ml
import pytorch_lightning as pl
import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.callbacks import RichProgressBar, ProgressBar

from lightning.model import XvecTDNN
from lightning.datamodules import get_datamodule, XvecDataModule
from lightning.systems import get_system
from dataset import XvecDataset

quiet = False
os.environ["COMET_LOGGING_CONSOLE"] = "DEBUG"

TRAINER_CONFIG = {
    "accelerator": "gpu",
    "devices": [0],
    "strategy": None,
    "auto_select_gpus": True,
    "limit_train_batches": 1.0,  # Useful for fast experiment
    "deterministic": True,
    "profiler": "simple",
}


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, train_config = configs

    trainer_training_config = {
        'max_steps': train_config["step"]["total_step"],
        'log_every_n_steps': train_config["step"]["log_step"],
        'weights_save_path': train_config["path"]["ckpt_path"],
        'gradient_clip_val': train_config["optimizer"]["grad_clip_thresh"],
        'accumulate_grad_batches': train_config["optimizer"]["grad_acc_step"],
        'resume_from_checkpoint': None,
    }

    if args.stage == 'train':
        log_dir = tempfile.mkdtemp()
        result_dir = tempfile.mkdtemp()
    else:
        assert args.exp_key is not None
        log_dir = tempfile.mkdtemp()
        result_dir = tempfile.mkdtemp()

    # Get dataset
    datamodule = XvecDataModule(preprocess_config, train_config, split="speaker")
    # Get model
    model = XvecTDNN(
        len(XvecDataset.accent_map),    # accent_map/region_map
        p_dropout=0
    )

    if args.stage == 'train':
        # datamodule.setup("fit")
        # Get model
        system = get_system("xvec")(model, train_config["step"]["total_step"])
        # Train
        trainer = pl.Trainer(
            **TRAINER_CONFIG, **trainer_training_config,
            enable_progress_bar=True, callbacks=[RichProgressBar()],
        )
        pl.seed_everything(43, True)
        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--preprocess_config", type=str, help="path to preprocess.yaml",
        default='config/preprocess/VCTK-xvec.yaml',
    )
    parser.add_argument(
        "-t", "--train_config", type=str, nargs='+', help="path to train.yaml",
        default=['config/train/xvec.yaml', 'config/train/VCTK.yaml'],
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open(args.train_config[0], "r"), Loader=yaml.FullLoader
    )
    train_config.update(
        yaml.load(open(args.train_config[1], "r"), Loader=yaml.FullLoader)
    )
    configs = (preprocess_config, train_config)

    main(args, configs)
