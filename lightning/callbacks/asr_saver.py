import os
import pandas as pd
import pytorch_lightning as pl
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from tqdm import tqdm
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.base import merge_dicts
from pytorch_lightning.utilities import rank_zero_only

from lightning.utils import asr_loss2dict as loss2dict
from text.define import LANG_ID2SYMBOLS


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

        self.re_id = False
        increment = 0
        self.re_id_increment = {}
        for k, v in LANG_ID2SYMBOLS.items():
            self.re_id_increment[k] = increment
            increment += len(v)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        loss = outputs['losses']
        output = outputs['output']
        _batch = outputs['_batch']  # batch or qry_batch
        lang_id = outputs['lang_id']
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
            mask = (_batch[3] != 0)
            acc = ((_batch[3] == output.argmax(dim=2)) * mask).sum() / mask.sum()
            pl_module.log_dict({"Train/Acc": acc.item()})

            # log asr results
            sentence = _batch[3][0]
            gt_sentence, pred_sentence = [], []
            for (gt_id, pred_id) in zip(sentence, output[0].argmax(dim=1)):
                if gt_id == 0:
                    break
                if self.re_id:
                    gt_id = int(gt_id) - self.re_id_increment[lang_id]
                    pred_id = int(pred_id) - self.re_id_increment[lang_id]
                gt_sentence.append(LANG_ID2SYMBOLS[lang_id][gt_id])
                pred_sentence.append(LANG_ID2SYMBOLS[lang_id][pred_id])
            
            self.log_text(logger, "Train/GT: " + ", ".join(gt_sentence), step)
            self.log_text(logger, "Train/Pred: " + ", ".join(pred_sentence), step)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_loss_dicts = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        loss = outputs['losses']
        output = outputs['output']
        _batch = outputs['_batch']  # batch or qry_batch
        lang_id = outputs['lang_id']
        
        step = pl_module.global_step + 1
        assert len(list(pl_module.logger)) == 1
        logger = pl_module.logger[0]

        loss_dict = loss2dict(loss)
        self.val_loss_dicts.append(loss_dict)

        # Log loss for each sample to csv files
        self.log_csv("Validation", step, 0, loss_dict)

        # Log asr results to logger + calculate acc
        if batch_idx == 0 and pl_module.local_rank == 0:

            # log asr results
            sentence = _batch[3][0]
            gt_sentence, pred_sentence = [], []
            for (gt_id, pred_id) in zip(sentence, output[0].argmax(dim=1)):
                if gt_id == 0:
                    break
                if self.re_id:
                    gt_id = int(gt_id) - self.re_id_increment[lang_id]
                    pred_id = int(pred_id) - self.re_id_increment[lang_id]
                gt_sentence.append(LANG_ID2SYMBOLS[lang_id][gt_id])
                pred_sentence.append(LANG_ID2SYMBOLS[lang_id][pred_id])
            
            self.log_text(logger, "Val/GT: " + ", ".join(gt_sentence), step)
            self.log_text(logger, "Val/Pred: " + ", ".join(pred_sentence), step)

    def log_csv(self, stage, step, basename, loss_dict):
        if stage in ("Training", "Validation"):
            log_dir = os.path.join(self.log_dir, "csv", stage)
        else:
            log_dir = os.path.join(self.result_dir, "csv", stage)
        os.makedirs(log_dir, exist_ok=True)
        csv_file_path = os.path.join(log_dir, f"{basename}.csv")

        df = pd.DataFrame(loss_dict, columns=CSV_COLUMNS, index=[step])
        df.to_csv(csv_file_path, mode='a', header=not os.path.exists(csv_file_path), index=True, index_label="Step")

    def log_text(self, logger, text, step):
        if isinstance(logger, pl.loggers.CometLogger):
            logger.experiment.log_text(
                text=text,
                step=step,
            )
