import os
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.loggers.base import merge_dicts
from pytorch_lightning.utilities import rank_zero_only

from .base_saver import BaseSaver
from lightning.utils import loss2dict
from lightning.callbacks.utils import synth_one_sample_with_target, synth_samples


CSV_COLUMNS = ["Total Loss", "Mel Loss", "Mel-Postnet Loss", "Pitch Loss", "Energy Loss", "Duration Loss"]
COL_SPACE = [len(col) for col in ["200000", "Validation"]+CSV_COLUMNS]  # max step: 200000, longest stage: validation


class Saver(BaseSaver):

    def __init__(self, preprocess_config, log_dir=None, result_dir=None):
        super().__init__(log_dir, result_dir)
        self.preprocess_config = preprocess_config
        self.sr = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        
        self.val_loss_dicts = []
        self.log_loss_dicts = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        loss = outputs['losses']
        output = outputs['output']
        _batch = outputs['_batch']

        step = pl_module.global_step + 1
        assert len(list(pl_module.logger)) == 1
        logger = pl_module.logger[0]
        vocoder = pl_module.vocoder

        # Synthesis one sample and log to CometLogger
        if step % pl_module.train_config["step"]["synth_step"] == 0 and pl_module.local_rank == 0:
            metadata = {'ids': batch[0]}
            fig, wav_reconstruction, wav_prediction, basename = synth_one_sample_with_target(
                _batch, output, vocoder, self.preprocess_config
            )
            self.log_figure(logger, "Training", step, basename, "", fig)
            self.log_audio(logger, "Training", step, basename, "reconstructed", wav_reconstruction, self.sr, metadata)
            self.log_audio(logger, "Training", step, basename, "synthesized", wav_prediction, self.sr, metadata)
            plt.close(fig)

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

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_loss_dicts = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        loss = outputs['losses']
        output = outputs['output']
        _batch = outputs['_batch']
        synth_output = outputs['synth']
        
        step = pl_module.global_step + 1
        assert len(list(pl_module.logger)) == 1
        logger = pl_module.logger[0]
        vocoder = pl_module.vocoder

        loss_dict = loss2dict(loss)
        self.val_loss_dicts.append(loss_dict)

        # Log loss for each sample to csv files
        self.save_csv("Validation", step, 0, loss_dict)

        figure_dir = os.path.join(self.result_dir, "figure")
        audio_dir = os.path.join(self.result_dir, "audio")
        os.makedirs(figure_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)

        # Log figure/audio to logger + save audio
        # One smaple for the first two batches, so synthesize two samples in total.
        if batch_idx == 0 and pl_module.local_rank == 0:
            metadata = {'ids': batch[0]}
            fig, wav_reconstruction, wav_prediction, basename = synth_one_sample_with_target(
                _batch, output, vocoder, self.preprocess_config
            )
            self.log_figure(logger, "Validation", step, basename, "", fig)
            self.log_audio(logger, "Validation", step, basename, "reconstructed", wav_reconstruction, self.sr, metadata)
            self.log_audio(logger, "Validation", step, basename, "synthesized", wav_prediction, self.sr, metadata)
            plt.close(fig)

            synth_samples(_batch, synth_output, vocoder, self.preprocess_config, figure_dir, audio_dir, f"FTstep_{step}")

    def on_validation_epoch_end(self, trainer, pl_module):
        loss_dict = merge_dicts(self.val_loss_dicts)
        step = pl_module.global_step + 1

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
