import argparse
import os

import comet_ml
import pytorch_lightning as pl
import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pytorch_lightning.profiler import AdvancedProfiler

from config.comet import COMET_CONFIG
from lightning.datamodules import get_datamodule
from lightning.systems import get_system

quiet = False
if quiet:
    # NOTSET/DEBUG/INFO/WARNING/ERROR/CRITICAL
    os.environ["COMET_LOGGING_CONSOLE"] = "ERROR"
    import warnings
    warnings.filterwarnings("ignore")
    import logging
    # configure logging at the root level of lightning
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINER_CONFIG = {
    "gpus": -1 if torch.cuda.is_available() else None,
    "strategy": "ddp" if torch.cuda.is_available() else None,
    "auto_select_gpus": True,
    "limit_train_batches": 1.0,  # Useful for fast experiment
    "deterministic": True,
    "process_position": 1,
    "profiler": 'simple',
}


def main(args, configs):
    print("Prepare training ...")

    preprocess_configs, model_config, train_config, algorithm_config = configs

    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)

    # Checkpoint for resume training or testing
    ckpt_file = None
    if args.exp_key is not None:
        ckpt_file = os.path.join(
            'output/ckpt/LibriTTS', COMET_CONFIG["project_name"],
            args.exp_key, 'checkpoints', args.ckpt_file
        )

    trainer_training_config = {
        'max_steps': train_config["step"]["total_step"],
        'log_every_n_steps': train_config["step"]["log_step"],
        'weights_save_path': train_config["path"]["ckpt_path"],
        'gradient_clip_val': train_config["optimizer"]["grad_clip_thresh"],
        'accumulate_grad_batches': train_config["optimizer"]["grad_acc_step"],
        'resume_from_checkpoint': ckpt_file,
    }
    if algorithm_config["type"] == 'imaml':
        # should manually clip grad
        del trainer_training_config['gradient_clip_val']

    if args.stage == 'train':
        # Init logger
        comet_logger = pl.loggers.CometLogger(
            save_dir=os.path.join(train_config["path"]["log_path"], "meta"),
            experiment_key=args.exp_key,
            experiment_name=algorithm_config["name"],
            **COMET_CONFIG
        )
        comet_logger.log_hyperparams({
            "preprocess_config": preprocess_configs,
            "model_config": model_config,
            "train_config": train_config,
            "algorithm_config": algorithm_config,
        })
        loggers = [comet_logger]
        log_dir = os.path.join(comet_logger._save_dir, comet_logger.version)
        result_dir = os.path.join(
            train_config['path']['result_path'], comet_logger.version
        )
    else:
        assert args.exp_key is not None
        log_dir = os.path.join(
            train_config["path"]["log_path"], "meta", args.exp_key
        )
        result_dir = os.path.join(
            train_config['path']['result_path'], args.exp_key, algorithm_config["name"]
        )

    # Get dataset
    datamodule = get_datamodule(algorithm_config["type"])(
        preprocess_configs, train_config, algorithm_config, log_dir, result_dir
    )

    if args.stage == 'train':
        # Get model
        system = get_system(algorithm_config["type"])
        model = system(
            preprocess_configs[0], model_config, train_config, algorithm_config,
            log_dir, result_dir
        )
        # Train
        trainer = pl.Trainer(
            logger=loggers, **TRAINER_CONFIG, **trainer_training_config
        )
        pl.seed_everything(43, True)
        trainer.fit(model, datamodule=datamodule)

    elif args.stage == 'test' or args.stage == 'predict':
        # Get model
        system = get_system(algorithm_config["type"])
        model = system.load_from_checkpoint(
            ckpt_file,
            preprocess_config=preprocess_configs[0],
            model_config=model_config,
            train_config=train_config,
            algorithm_config=algorithm_config,
            log_dir=log_dir, result_dir=result_dir,
            strict=False,
        )
        # Test
        trainer = pl.Trainer(**TRAINER_CONFIG)
        trainer.test(model, datamodule=datamodule)

    elif args.stage == 'debug':
        del datamodule
        datamodule = get_datamodule("base")(
            preprocess_configs, train_config, algorithm_config, log_dir, result_dir
        )
        datamodule.setup('test')
        for _ in tqdm(datamodule.test_dataset, desc="test_dataset"):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--preprocess_config", type=str, nargs='+', help="path to preprocess.yaml",
        default=['config/preprocess/miniLibriTTS.yaml'],
        # default=['config/preprocess/LibriTTS.yaml'],
    )
    parser.add_argument(
        "-m", "--model_config", type=str, help="path to model.yaml",
        default='config/model/dev.yaml',
        # default='config/model/base.yaml',
    )
    parser.add_argument(
        "-t", "--train_config", type=str, nargs='+', help="path to train.yaml",
        default=['config/train/dev.yaml', 'config/train/miniLibriTTS.yaml'],
        # default=['config/train/base.yaml', 'config/train/LibriTTS.yaml'],
    )
    parser.add_argument(
        "-a", "--algorithm_config", type=str, help="path to algorithm.yaml",
        default='config/algorithm/dev.yaml',
    )
    parser.add_argument(
        "-e", "--exp_key", type=str, help="experiment key",
        default=None,
    )
    parser.add_argument(
        "-c", "--ckpt_file", type=str, help="ckpt file name",
        default="last.ckpt",
    )
    parser.add_argument(
        "-s", "--stage", type=str, help="stage (train/val/test/predict)",
        default="train",
    )
    args = parser.parse_args()

    # Read Config
    preprocess_configs = [
        yaml.load(open(path, "r"), Loader=yaml.FullLoader)
        for path in args.preprocess_config
    ]
    model_config = yaml.load(
        open(args.model_config, "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open(args.train_config[0], "r"), Loader=yaml.FullLoader
    )
    train_config.update(
        yaml.load(open(args.train_config[1], "r"), Loader=yaml.FullLoader)
    )
    algorithm_config = yaml.load(
        open(args.algorithm_config, "r"), Loader=yaml.FullLoader
    )
    configs = (preprocess_configs, model_config, train_config, algorithm_config)

    main(args, configs)
