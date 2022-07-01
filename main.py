import traceback
import warnings
import logging
import sys

import argparse
import os

import comet_ml
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.comet import COMET_CONFIG
from lightning.datamodules import get_datamodule
from lightning.systems import get_system


quiet = None
if quiet is not None:
    if quiet:
        # NOTSET/DEBUG/INFO/WARNING/ERROR/CRITICAL
        os.environ["COMET_LOGGING_CONSOLE"] = "ERROR"
        warnings.filterwarnings("ignore")
        # configure logging at the root level of lightning
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    else:
        def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

            log = file if hasattr(file,'write') else sys.stderr
            traceback.print_stack(file=log)
            log.write(warnings.formatwarning(message, category, filename, lineno, line))
        warnings.showwarning = warn_with_traceback


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAINER_CONFIG = {
    "accelerator": "gpu",
    "strategy": "ddp" if torch.cuda.is_available() else None,
    "auto_select_gpus": True,
    "limit_train_batches": 1.0,  # Useful for fast experiment
    # "deterministic": True,
    "process_position": 1,
    "profiler": "simple",
}

def init_logger(args, configs):
    preprocess_configs, model_config, train_config, algorithm_config = configs

    trainer_training_config = {
        'max_steps': train_config["step"]["total_step"],
        'log_every_n_steps': train_config["step"]["log_step"],
        'check_val_every_n_epoch': int(train_config["step"]["val_step"] //
                                       train_config["step"]["log_step"]),
        'weights_save_path': train_config["path"]["ckpt_path"],
        'gradient_clip_val': train_config["optimizer"]["grad_clip_thresh"],
        'accumulate_grad_batches': train_config["optimizer"]["grad_acc_step"],
    }

    if algorithm_config["type"] == 'imaml':
        # should manually clip grad
        del trainer_training_config['gradient_clip_val']

    if args.stage == "train":
        # Init logger
        comet_logger = CometLogger(
            save_dir=os.path.join(train_config["path"]["log_path"],
                                  COMET_CONFIG["project_name"]),
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
        logger = comet_logger
        log_dir = os.path.join(comet_logger._save_dir, comet_logger.version)
        result_dir = os.path.join(
            train_config['path']['result_path'], COMET_CONFIG["project_name"],
            comet_logger.version
        )
        trainer = pl.Trainer(
            logger=logger,
            **TRAINER_CONFIG, **trainer_training_config
        )
    elif args.stage == "transfer":
        # Init logger
        comet_logger = CometLogger(
            save_dir=os.path.join(train_config["path"]["log_path"],
                                  COMET_CONFIG["project_name"]),
            experiment_key=None,
            experiment_name=algorithm_config["name"],
            **COMET_CONFIG
        )
        comet_logger.log_hyperparams({
            "preprocess_config": preprocess_configs,
            "model_config": model_config,
            "train_config": train_config,
            "algorithm_config": algorithm_config,
        })
        logger = comet_logger
        log_dir = os.path.join(comet_logger._save_dir, comet_logger.version)
        result_dir = os.path.join(
            train_config['path']['result_path'], COMET_CONFIG["project_name"],
            comet_logger.version
        )
        trainer = pl.Trainer(
            logger=logger, **TRAINER_CONFIG, **trainer_training_config,
        )
    elif args.stage in {"dev.train", "dev.transfer"}:
        # Init logger
        log_dir = f"output/log/{algorithm_config['name']}"
        result_dir = f"output/result/{algorithm_config['name']}"
        TRAINER_CONFIG["devices"] = [0]
        del trainer_training_config['weights_save_path']
        trainer = pl.Trainer(
            default_root_dir=f"output/{algorithm_config['name']}",
            **TRAINER_CONFIG, **trainer_training_config
        )
    elif args.stage in {"dev.val"}:
        # Init logger
        log_dir = f"output/{algorithm_config['name']}/log"
        result_dir = f"output/{algorithm_config['name']}/result"
        del TRAINER_CONFIG["auto_select_gpus"], TRAINER_CONFIG["strategy"]
        TRAINER_CONFIG["devices"] = [0]
        trainer = pl.Trainer(
            default_root_dir=f"output/{algorithm_config['name']}",
            **TRAINER_CONFIG,
        )
    elif args.stage == "val":
        log_dir = os.path.join(
            train_config["path"]["log_path"], COMET_CONFIG["project_name"],
            args.exp_key
        )
        result_dir = os.path.join(
            train_config['path']['result_path'], COMET_CONFIG["project_name"],
            args.exp_key, algorithm_config["name"]
        )
        del TRAINER_CONFIG["auto_select_gpus"], TRAINER_CONFIG["strategy"]
        TRAINER_CONFIG["devices"] = [0]
        trainer = pl.Trainer(
            logger=TensorBoardLogger(
                save_dir=log_dir, name=algorithm_config["name"]),
            **TRAINER_CONFIG,
        )
    else:
        assert args.exp_key is not None
        log_dir = os.path.join(
            train_config["path"]["log_path"], COMET_CONFIG["project_name"],
            args.exp_key
        )
        result_dir = os.path.join(
            train_config['path']['result_path'], COMET_CONFIG["project_name"],
            args.exp_key, algorithm_config["name"]
        )
        trainer = pl.Trainer(**TRAINER_CONFIG)
    return log_dir, result_dir, trainer


def main(args, configs):
    print("Prepare training ...")

    preprocess_configs, model_config, train_config, algorithm_config = configs

    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)

    log_dir, result_dir, trainer = init_logger(args, configs)

    # Get dataset
    datamodule = get_datamodule(algorithm_config["type"].split('.')[0])(
        preprocess_configs, train_config, algorithm_config,
        stage=args.stage.split('.')[-1],
    )

    system_config = {
        "preprocess_config": preprocess_configs[0],
        "model_config": model_config,
        "train_config": train_config,
        "algorithm_config": algorithm_config,
    }

    if args.stage in {"train", "dev.train"}:
        # Get model
        system = get_system(algorithm_config["type"])(
            **system_config,
            log_dir=log_dir, result_dir=result_dir,
        )
        # Train
        pl.seed_everything(43, True)
        print(args.ckpt_file)
        trainer.fit(system, datamodule=datamodule, ckpt_path=args.ckpt_file)

    elif args.stage in {"transfer", "dev.transfer"}:
        # Get model
        system = get_system(algorithm_config["type"]).load_from_checkpoint(
            args.pretrain_ckpt_path,
            **system_config,
            log_dir=log_dir, result_dir=result_dir,
            strict=False,
        )
        # Train
        pl.seed_everything(43, True)
        trainer.fit(system, datamodule=datamodule, ckpt_path=args.ckpt_file)

    elif args.stage in {"val", "dev.val"}:
        # Get model
        system = get_system(algorithm_config["type"])(
            **system_config,
            log_dir=log_dir, result_dir=result_dir,
        )
        # Trainer

        # ckpt_dir = os.path.dirname(args.ckpt_file)
        # for epoch in range(19, 99, 10):
        #     ckpt_file = os.path.join(ckpt_dir, f"epoch={epoch}-step={epoch}999.ckpt")
        #     system.load_state_dict(torch.load(ckpt_file), strict=False)
        #     # Val
        #     trainer.validate(system, datamodule=datamodule, ckpt_path=ckpt_file)
        system.load_state_dict(torch.load(args.ckpt_file), strict=False)
        # Val
        trainer.validate(system, datamodule=datamodule, ckpt_path=args.ckpt_file)

    elif args.stage == "test" or args.stage == "predict":
        # Get model
        system = get_system(algorithm_config["type"]).load_from_checkpoint(
            args.ckpt_file,
            **system_config,
            log_dir=log_dir, result_dir=result_dir,
            strict=False,
        )
        # Test
        trainer.test(system, ckpt_path=args.ckpt_file, datamodule=datamodule)


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
        "-s", "--stage", type=str, help="stage (train/val/test/predict/transfer)",
        default="train",
    )
    args = parser.parse_args()

    # Read Config
    algorithm_config = yaml.load(
        open(args.algorithm_config, "r"), Loader=yaml.FullLoader
    )
    if "parser_args" in algorithm_config:
        args_config = algorithm_config["parser_args"]
        preprocess_configs = [
            yaml.load(open(path, "r"), Loader=yaml.FullLoader)
            for path in args_config["preprocess_config"]
        ]
        model_config = yaml.load(
            open(args_config["model_config"], "r"), Loader=yaml.FullLoader
        )
        train_config = yaml.load(
            open(args_config["train_config"][0], "r"), Loader=yaml.FullLoader
        )
        train_config.update(
            yaml.load(open(args_config["train_config"][1], "r"), Loader=yaml.FullLoader)
        )
        args.exp_key = args_config["exp_key"]
        if args.ckpt_file == "last.ckpt":
            args.ckpt_file = args_config["ckpt_file"]
        args.pretrain_ckpt_path = args_config.get("pretrain_ckpt_path", None)
        args.stage = args_config["stage"]

        if "ckpt_path" in args_config:
            args.ckpt_file = args_config["ckpt_path"]
        else:
            # Checkpoint for resume training or testing
            ckpt_file = None
            if args.exp_key is not None:
                ckpt_file = os.path.join(
                    'output/ckpt/LibriTTS', COMET_CONFIG["project_name"],
                    args.exp_key, 'checkpoints', args.ckpt_file
                )
            args.ckpt_file = ckpt_file
    else:
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

        # Checkpoint for resume training or testing
        ckpt_file = None
        if args.exp_key is not None:
            ckpt_file = os.path.join(
                'output/ckpt/LibriTTS', COMET_CONFIG["project_name"],
                args.exp_key, 'checkpoints', args.ckpt_file
            )
        args.ckpt_file = ckpt_file
    configs = (preprocess_configs, model_config, train_config, algorithm_config)

    main(args, configs)
