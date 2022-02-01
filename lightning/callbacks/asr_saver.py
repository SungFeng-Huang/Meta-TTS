import os
import pandas as pd
import pytorch_lightning as pl
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from tqdm import tqdm
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.base import merge_dicts

from lightning.utils import asr_loss2dict as loss2dict


CSV_COLUMNS = ["Total Loss"]
COL_SPACE = [len(col) for col in ["200000", "Validation"]+CSV_COLUMNS]  # max step: 200000, longest stage: validation


class Saver(Callback):

    def __init__(self, preprocess_config, log_dir=None, result_dir=None):
        super().__init__()
        self.preprocess_config = preprocess_config

        self.log_dir = log_dir
        self.result_dir = result_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        print("Log directory:", self.log_dir)

        self.val_loss_dicts = []
        self.log_loss_dicts = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        loss = outputs['losses']
        output = outputs['output']
        _batch = outputs['_batch']  # batch or qry_batch
        step = pl_module.global_step + 1
        assert len(list(pl_module.logger)) == 1
        logger = pl_module.logger[0]

        # Log message to log.txt and print to stdout
        if step % trainer.log_every_n_steps == 0 and pl_module.local_rank == 0:
            loss_dict = loss2dict(loss)
            loss_dict.update({"Step": step, "Stage": "Training"})
            df = pd.DataFrame([loss_dict], columns=["Step", "Stage"]+CSV_COLUMNS)
            if len(self.log_loss_dicts)==0:
                tqdm.write(df.to_string(header=True, index=False, col_space=COL_SPACE))
            else:
                tqdm.write(df.to_string(header=True, index=False, col_space=COL_SPACE).split('\n')[-1])
            self.log_loss_dicts.append(loss_dict)

            # calculate acc

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_loss_dicts = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        loss = outputs['losses']
        output = outputs['output']
        _batch = outputs['_batch']  # batch or qry_batch
        
        step = pl_module.global_step + 1
        assert len(list(pl_module.logger)) == 1
        logger = pl_module.logger[0]

        loss_dict = loss2dict(loss)
        self.val_loss_dicts.append(loss_dict)

        # Log loss for each sample to csv files
        sup_ids = batch[0]
        # SQids = f"{'-'.join(sup_ids)}.{'-'.join(qry_ids)}"
        # task_id = trainer.datamodule.val_SQids2Tid[SQids]
        self.log_csv("Validation", step, 0, loss_dict)

        # Log asr results to logger + calculate acc
        if batch_idx == 0 and pl_module.local_rank == 0:
            metadata = {'sup_ids': sup_ids}

    def on_validation_epoch_end(self, trainer, pl_module):
        # if pl_module.global_step > 0:
        if True:
            loss_dict = merge_dicts(self.val_loss_dicts)
            step = pl_module.global_step+1

            # Log total loss to log.txt and print to stdout
            loss_dict.update({"Step": step, "Stage": "Validation"})
            # To stdout
            df = pd.DataFrame([loss_dict], columns=["Step", "Stage"]+CSV_COLUMNS)
            if len(self.log_loss_dicts)==0:
                tqdm.write(df.to_string(header=True, index=False, col_space=COL_SPACE))
            else:
                tqdm.write(df.to_string(header=True, index=False, col_space=COL_SPACE).split('\n')[-1])
            # To file
            self.log_loss_dicts.append(loss_dict)
            log_file_path = os.path.join(self.log_dir, 'log.txt')
            df = pd.DataFrame(self.log_loss_dicts, columns=["Step", "Stage"]+CSV_COLUMNS).set_index("Step")
            df.to_csv(log_file_path, mode='a', header=not os.path.exists(log_file_path), index=True)
            # Reset
            self.log_loss_dicts = []

    def log_csv(self, stage, step, basename, loss_dict):
        if stage in ("Training", "Validation"):
            log_dir = os.path.join(self.log_dir, "csv", stage)
        else:
            log_dir = os.path.join(self.result_dir, "csv", stage)
        os.makedirs(log_dir, exist_ok=True)
        csv_file_path = os.path.join(log_dir, f"{basename}.csv")

        df = pd.DataFrame(loss_dict, columns=CSV_COLUMNS, index=[step])
        df.to_csv(csv_file_path, mode='a', header=not os.path.exists(csv_file_path), index=True, index_label="Step")
