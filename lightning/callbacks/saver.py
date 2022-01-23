import os
import pandas as pd
import pytorch_lightning as pl
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from tqdm import tqdm
from scipy.io import wavfile
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.base import merge_dicts
from pytorch_lightning.utilities import rank_zero_only

from utils.tools import expand, plot_mel
from lightning.utils import LightningMelGAN, loss2dict, dict2loss, loss2str, dict2str
from lightning.callbacks.utils import synth_one_sample_with_target, recon_samples, synth_samples


CSV_COLUMNS = ["Total Loss", "Mel Loss", "Mel-Postnet Loss", "Pitch Loss", "Energy Loss", "Duration Loss"]
COL_SPACE = [len(col) for col in ["200000", "Validation"]+CSV_COLUMNS]  # max step: 200000, longest stage: validation


class Saver(Callback):
    """
    """

    def __init__(self, preprocess_config, log_dir=None, result_dir=None):
        super().__init__()
        self.preprocess_config = preprocess_config

        self.log_dir = log_dir
        self.result_dir = result_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        print("Log directory:", self.log_dir)
        print("Result directory:", self.result_dir)

        self.val_loss_dicts = []
        self.log_loss_dicts = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        loss = outputs['losses']
        output = outputs['output']
        _batch = outputs['_batch']  # batch or qry_batch
        step = pl_module.global_step+1
        assert len(list(pl_module.logger)) == 1
        logger = pl_module.logger[0]
        vocoder = pl_module.vocoder

        # Synthesis one sample and log to CometLogger
        if step % pl_module.train_config["step"]["synth_step"] == 0 and pl_module.local_rank == 0:
            metadata = {'sup_ids': batch[0][0][0][0]}
            fig, wav_reconstruction, wav_prediction, basename = synth_one_sample_with_target(
                _batch, output, vocoder, self.preprocess_config
            )
            self.log_figure(logger, "Training", step, basename, "", fig)
            self.log_audio(logger, "Training", step, basename, "reconstructed", wav_reconstruction, metadata)
            self.log_audio(logger, "Training", step, basename, "synthesized", wav_prediction, metadata)
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
        _batch = outputs['_batch']  # batch or qry_batch
        
        step = pl_module.global_step+1
        assert len(list(pl_module.logger)) == 1
        logger = pl_module.logger[0]
        vocoder = pl_module.vocoder

        loss_dict = loss2dict(loss)
        self.val_loss_dicts.append(loss_dict)

        # Log loss for each sample to csv files
        sup_ids = batch[0][0][0][0]
        qry_ids = batch[0][1][0][0]
        SQids = f"{'-'.join(sup_ids)}.{'-'.join(qry_ids)}"
        task_id = trainer.datamodule.val_SQids2Tid[SQids]
        self.log_csv("Validation", step, task_id, loss_dict)

        # Log figure/audio to logger + save audio
        if batch_idx == 0 and pl_module.local_rank == 0:
            metadata = {'sup_ids': sup_ids}
            fig, wav_reconstruction, wav_prediction, basename = synth_one_sample_with_target(
                _batch, output, vocoder, self.preprocess_config
            )
            self.log_figure(logger, "Validation", step, basename, "", fig)
            self.log_audio(logger, "Validation", step, basename, "reconstructed", wav_reconstruction, metadata)
            self.log_audio(logger, "Validation", step, basename, "synthesized", wav_prediction, metadata)
            plt.close(fig)

    def on_validation_epoch_end(self, trainer, pl_module):
        # if pl_module.global_step > 0:
        if True:
            loss_dict = merge_dicts(self.val_loss_dicts)
            step = pl_module.global_step+1

            # Log total loss to log.txt and print to stdout
            loss_dict.update({"Step": step, "Stage": "Validation"})
            # To stdout
            df = pd.DataFrame([loss_dict], columns=["Step", "Stage"]+CSV_COLUMNS)
            if pl_module.local_rank == 0:
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

    def on_test_batch_end(self, trainer, pl_module, all_outputs, batch, batch_idx, dataloader_idx):
        global_step = getattr(pl_module, 'test_global_step', pl_module.global_step)
        adaptation_steps = pl_module.adaptation_steps           # log/save period
        test_adaptation_steps = pl_module.test_adaptation_steps # total fine-tune steps
        vocoder = pl_module.vocoder

        sup_ids = batch[0][0][0][0]
        qry_ids = batch[0][1][0][0]
        SQids = f"{'-'.join(sup_ids)}.{'-'.join(qry_ids)}"
        _task_id = trainer.datamodule.test_SQids2Tid[SQids]

        for i, outputs in enumerate(all_outputs):
            if len(all_outputs) == 1:
                task_id = _task_id
            else:
                task_id = f"{_task_id}_{i}"

            figure_dir = os.path.join(self.result_dir, "figure", "Testing", f"step_{global_step}", task_id)
            audio_dir = os.path.join(self.result_dir, "audio", "Testing", f"step_{global_step}", task_id)
            log_dir = os.path.join(self.result_dir, "csv", "Testing", f"step_{global_step}")
            os.makedirs(figure_dir, exist_ok=True)
            os.makedirs(audio_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)
            csv_file_path = os.path.join(log_dir, f"{task_id}.csv")

            loss_dicts = []
            _batch = outputs["_batch"]

            for ft_step in range(0, test_adaptation_steps+1, adaptation_steps):
                if ft_step == 0:
                    predictions = outputs[f"step_{ft_step}"]["recon"]["output"]
                    recon_samples(
                        _batch, predictions, vocoder, self.preprocess_config,
                        figure_dir, audio_dir
                    )

                if "recon" in outputs[f"step_{ft_step}"]:
                    valid_error = outputs[f"step_{ft_step}"]["recon"]["losses"]
                    loss_dicts.append({"Step": ft_step, **loss2dict(valid_error)})

                if "synth" in outputs[f"step_{ft_step}"]:
                    predictions = outputs[f"step_{ft_step}"]["synth"]["output"]
                    synth_samples(
                        _batch, predictions, vocoder, self.preprocess_config,
                        figure_dir, audio_dir, f"step_{global_step}-FTstep_{ft_step}"
                    )

            df = pd.DataFrame(loss_dicts, columns=["Step"] + CSV_COLUMNS).set_index("Step")
            df.to_csv(csv_file_path, mode='a', header=True, index=True)

    def save_audio(self, stage, step, basename, tag, audio):
        """
        Parameters:
            stage (str): {"Training", "Validation", "Testing"}.
            step (int): Current step.
            basename (str): Audio index (original filename).
            tag (str): {"reconstructed", "synthesized"}.
            audio (numpy): Audio waveform.
        """
        sample_rate = self.preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        save_dir = os.path.join(self.log_dir, "audio", stage)
        filename = f"step_{step}_{basename}_{tag}.wav"

        os.makedirs(save_dir, exist_ok=True)
        wavfile.write(os.path.join(save_dir, filename), sample_rate, audio)

    def log_text(self, stage, loss_dict, print_fn=print, max_steps=200000, header=False):
        if header:
            message = f"{'Step':^{len(str(max_steps))}}, {'Stage':^{len('Validation')}}, "
            pl_module.print(message + ", ".join(CSV_COLUMNS))
        message = f"{loss_dict['Step']:{len(str(max_steps))}}, {loss_dict['Stage']:<{len('Validation')}}, "
        print_fn(message + ", ".join([f"{loss_dict[col]}:{len(col)}.4f" for col in CSV_COLUMNS]))

    def log_csv(self, stage, step, basename, loss_dict):
        if stage in ("Training", "Validation"):
            log_dir = os.path.join(self.log_dir, "csv", stage)
        else:
            log_dir = os.path.join(self.result_dir, "csv", stage)
        os.makedirs(log_dir, exist_ok=True)
        csv_file_path = os.path.join(log_dir, f"{basename}.csv")

        df = pd.DataFrame(loss_dict, columns=CSV_COLUMNS, index=[step])
        df.to_csv(csv_file_path, mode='a', header=not os.path.exists(csv_file_path), index=True, index_label="Step")

    def log_figure(self, logger, stage, step, basename, tag, figure):
        """
        Parameters:
            logger (LightningLoggerBase): {pl.loggers.CometLogger, pl.loggers.TensorBoardLogger}.
            stage (str): {"Training", "Validation", "Testing"}.
            step (int): Current step.
            basename (str): Audio index (original filename).
            tag (str): {"reconstructed", "synthesized"}.
            figure (matplotlib.pyplot.figure):
        """
        figure_name = f"{stage}/step_{step}_{basename}"
        figure_name += f"_{tag}" if tag != "" else ""

        if isinstance(logger, pl.loggers.CometLogger):
            logger.experiment.log_figure(
                figure_name=figure_name,
                figure=figure,
                step=step,
            )
        elif isinstance(logger, pl.loggers.TensorBoardLogger):
            logger.experiment.add_figure(
                tag=figure_name,
                figure=figure,
                global_step=step,
            )
        else:
            print("Failed to log figure: not finding correct logger type")

    def log_audio(self, logger, stage, step, basename, tag, audio, metadata=None):
        """
        Parameters:
            logger (LightningLoggerBase): {pl.loggers.CometLogger, pl.loggers.TensorBoardLogger}.
            stage (str): {"Training", "Validation", "Testing"}.
            step (int): Current step.
            basename (str): Audio index (original filename).
            tag (str): {"reconstructed", "synthesized"}.
            audio (numpy): Audio waveform.
        """
        sample_rate = self.preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        self.save_audio(stage, step, basename, tag, audio)

        if isinstance(logger, pl.loggers.CometLogger):
            if metadata is None:
                metadata = {}
            metadata.update({'stage': stage, 'type': tag, 'id': basename})
            logger.experiment.log_audio(
                audio_data=audio / max(abs(audio)),
                sample_rate=sample_rate,
                file_name=f"{basename}_{tag}.wav",
                step=step,
                metadata=metadata,
            )
        elif isinstance(logger, pl.loggers.TensorBoardLogger):
            logger.experiment.add_audio(
                tag=f"{stage}/step_{step}_{basename}_{tag}",
                snd_tensor=audio / max(abs(audio)),
                global_step=step,
                sample_rate=sample_rate,
            )
        else:
            print("Failed to log audio: not finding correct logger type")

