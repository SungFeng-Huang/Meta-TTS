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
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import ProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor, GPUStatsMonitor, ModelCheckpoint
from pytorch_lightning.profiler import AdvancedProfiler

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss, FastSpeech2
from dataset import Dataset
from lightning.scheduler import get_scheduler
from lightning.optimizer import get_optimizer
from lightning.callbacks import GlobalProgressBar
from lightning.collate import get_single_collate
from lightning.utils import seed_all, EpisodicInfiniteWrapper


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    comet_logger = pl.loggers.CometLogger(
        save_dir=os.path.join(train_config["path"]["log_path"], "meta"),
        experiment_key=args.exp_key,
        log_code=True,
        log_graph=True,
        parse_args=True,
        log_env_details=True,
        log_git_metadata=True,
        log_git_patch=True,
        log_env_gpu=True,
        log_env_cpu=True,
        log_env_host=True,
    )
    comet_logger.log_hyperparams({
        "preprocess_config": preprocess_config,
        "train_config": train_config,
        "model_config": model_config,
    })
    loggers = [comet_logger]
    profiler = AdvancedProfiler(train_config["path"]["log_path"], 'profile.log')

    # Get dataset
    train_dataset = None
    val_dataset = None
    test_dataset = Dataset(
        f"{preprocess_config['subsets']['test']}.txt", preprocess_config, train_config, sort=False, drop_last=False
    )

    # Prepare model
    with seed_all(43):
        model = FastSpeech2(preprocess_config, model_config)
    optimizer = get_optimizer(model, model_config, train_config)
    scheduler = get_scheduler(optimizer, train_config)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    resume_ckpt = f'./output/ckpt/LibriTTS/meta-tts/{args.exp_key}/checkpoints/{args.ckpt_file}' #NOTE
    log_dir = os.path.join(comet_logger._save_dir, comet_logger.version)
    result_dir = os.path.join(train_config['path']['result_path'], comet_logger.version, args.algorithm)
    if args.algorithm == 'base' or args.algorithm == 'base_emb_vad':
        from lightning.baseline import BaselineSystem as System
    if args.algorithm == 'base_emb_va':
        from lightning.base_emb_va import BaselineEmbVASystem as System
    if args.algorithm == 'base_emb_d':
        from lightning.base_emb_d import BaselineEmbDecSystem as System
    if args.algorithm == 'base_emb':
        from lightning.base_emb import BaselineEmbSystem as System
    if args.algorithm == 'base_emb1_vad':
        from lightning.baseline import BaselineEmb1VADecSystem as System
    if args.algorithm == 'base_emb1_va':
        from lightning.base_emb_va import BaselineEmb1VASystem as System
    if args.algorithm == 'base_emb1_d':
        from lightning.base_emb_d import BaselineEmb1DecSystem as System
    if args.algorithm == 'base_emb1':
        from lightning.base_emb import BaselineEmb1System as System
    if args.algorithm == 'meta' or args.algorithm == 'meta_emb_vad':
        from lightning.anil import ANILSystem as System
    if args.algorithm == 'meta_emb_va':
        from lightning.anil_emb_va import ANILEmbVASystem as System
    if args.algorithm == 'meta_emb_d':
        from lightning.anil_emb_d import ANILEmbDecSystem as System
    if args.algorithm == 'meta_emb':
        from lightning.anil_emb import ANILEmbSystem as System
    if args.algorithm == 'meta_emb1_vad':
        from lightning.anil import ANILEmb1VADecSystem as System
    if args.algorithm == 'meta_emb1_va':
        from lightning.anil_emb_va import ANILEmb1VASystem as System
    if args.algorithm == 'meta_emb1_d':
        from lightning.anil_emb_d import ANILEmb1DecSystem as System
    if args.algorithm == 'meta_emb1':
        from lightning.anil_emb import ANILEmb1System as System
    try:
        system = System.load_from_checkpoint(
            resume_ckpt,
            strict=True,
            model=model,
            optimizer=optimizer,
            loss_func=Loss,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            scheduler=scheduler,
            configs=configs,
            vocoder=vocoder,
            log_dir=log_dir,
            result_dir=result_dir,
        )
    except Exception as e:
        system = System.load_from_checkpoint(
            resume_ckpt,
            strict=False,
            model=model,
            optimizer=optimizer,
            loss_func=Loss,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            scheduler=scheduler,
            configs=configs,
            vocoder=vocoder,
            log_dir=log_dir,
            result_dir=result_dir,
        )

    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    val_step = train_config["step"]["val_step"]

    callbacks = []
    checkpoint = ModelCheckpoint(
        monitor="val_Loss/total_loss",
        mode="min",
        save_top_k=-1,
        every_n_train_steps=save_step,
        save_last=True,
    )
    callbacks.append(checkpoint)

    outer_bar = GlobalProgressBar(process_position=1)
    inner_bar = ProgressBar(process_position=1)
    lr_monitor = LearningRateMonitor()
    gpu_monitor = GPUStatsMonitor(memory_utilization=True, gpu_utilization=True, intra_step_time=True, inter_step_time=True)
    callbacks.append(outer_bar)
    callbacks.append(inner_bar)
    callbacks.append(lr_monitor)
    callbacks.append(gpu_monitor)

    gpus = -1 if torch.cuda.is_available() else None
    distributed_backend = "ddp" if torch.cuda.is_available() else None

    trainer = pl.Trainer(
        max_steps=total_step,
        weights_save_path=train_config["path"]["ckpt_path"],
        callbacks=callbacks,
        logger=loggers,
        gpus=gpus,
        auto_select_gpus=True,
        distributed_backend=distributed_backend,
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=grad_clip_thresh,
        accumulate_grad_batches=grad_acc_step,
        resume_from_checkpoint=resume_ckpt,
        deterministic=True,
        log_every_n_steps=log_step,
        # profiler=profiler,
        profiler='simple',
    )
    print(resume_ckpt)
    trainer.test(system, ckpt_path=resume_ckpt, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        help="path to preprocess.yaml",
        default='config/LibriTTS/preprocess.yaml',
    )
    parser.add_argument(
        "-m", "--model_config", type=str, help="path to model.yaml",
        default='config/LibriTTS/model.yaml',
    )
    parser.add_argument(
        "-t", "--train_config", type=str, help="path to train.yaml",
        default='config/LibriTTS/train.yaml',
    )
    parser.add_argument(
        "-s", "--meta_batch_size", type=int, help="meta batch size",
        default=torch.cuda.device_count(),
    )
    parser.add_argument(
        "-e", "--exp_key", type=str, help="experiment key",
        default=None,
    )
    parser.add_argument(
        "-c", "--ckpt_file", type=str, help="ckpt file name",
        default=None,
    )
    parser.add_argument(
        "-d", "--dev", action="store_true", help="dev mode"
    )
    # base_emb* are all the same in training, and only different in validation/testing
    # *_emb1_* are for shared embedding, and *_emb_* are for embedding table
    parser.add_argument(
        "-a", "--algorithm", type=str, help="algorithm (required)",
        choices=[
            'base_emb_vad',
            'base_emb_va',
            'base_emb_d',
            'base_emb',
            'meta_emb_vad',
            'meta_emb_va',
            'meta_emb_d',
            'meta_emb',
            'base_emb1_vad',
            'base_emb1_va',
            'base_emb1_d',
            'base_emb1',
            'meta_emb1_vad',
            'meta_emb1_va',
            'meta_emb1_d',
            'meta_emb1',
        ]
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    train_config["meta"]["meta_batch_size"] = args.meta_batch_size
    if args.dev:
        train_config["step"]["total_step"] = 300
        train_config["step"]["synth_step"] = 100
        train_config["step"]["val_step"] = 100
        train_config["step"]["save_step"] = 100
        model_config["transformer"]["encoder_layer"] = 2
        model_config["transformer"]["decoder_layer"] = 2
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
