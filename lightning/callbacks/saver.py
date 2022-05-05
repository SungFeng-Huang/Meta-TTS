import os
import json
import pandas as pd
import pytorch_lightning as pl
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from tqdm import tqdm
from scipy.io import wavfile
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.base import merge_dicts, LoggerCollection
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

        self.vocoder = LightningMelGAN()
        self.vocoder.freeze()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs['losses']
        output = outputs['output']
        _batch = outputs['_batch']  # batch or qry_batch
        step = pl_module.global_step
        if isinstance(pl_module.logger, LoggerCollection):
            assert len(list(pl_module.logger)) == 1
            logger = pl_module.logger[0]
        else:
            logger = pl_module.logger
        self.vocoder.to(pl_module.device)

        # Synthesis one sample and log to CometLogger
        if pl_module.global_step % pl_module.train_config["step"]["synth_step"] == 0 and pl_module.local_rank == 0:
            metadata = {'sup_ids': batch[0][0][0][0]}
            fig, wav_reconstruction, wav_prediction, basename = synth_one_sample_with_target(
                _batch, output, self.vocoder, self.preprocess_config
            )
            figure_name = f"Training/step_{pl_module.global_step}_{basename}"
            self.log_figure(logger, figure_name, fig, pl_module.global_step)
            self.log_audio(logger, "Training", pl_module.global_step, basename, "reconstructed", wav_reconstruction, metadata)
            self.log_audio(logger, "Training", pl_module.global_step, basename, "synthesized", wav_prediction, metadata)
            plt.close(fig)

    def on_validation_batch_end(self, trainer, pl_module,
                                outputs, batch, batch_idx, dataloader_idx):
        if trainer.state.fn == "fit":
            # trainer.state.fn == TrainerFn.FITTING == "fit"
            return self._on_val_batch_end_for_fit(
                trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )
        elif trainer.state.fn == "validate":
            pass
            # return self._on_val_batch_end_for_validate(
            #     trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            # )

    def _on_val_batch_end_for_fit(self, trainer, pl_module,
                                  outputs, batch, batch_idx, dataloader_idx):
        loss = outputs['losses']
        output = outputs['output']
        _batch = outputs['_batch']  # batch or qry_batch

        logger = pl_module.logger

        self.vocoder.to(pl_module.device)

        # Log loss for each sample to csv files
        sup_ids = batch[0][0][0][0]
        qry_ids = batch[0][1][0][0]
        SQids = f"{'-'.join(sup_ids)}.{'-'.join(qry_ids)}"
        task_id = trainer.datamodule.val_SQids2Tid[SQids]
        self.log_csv("Validation", pl_module.global_step, task_id, loss2dict(loss))

        # Log figure/audio to logger + save audio
        if batch_idx == 0 and pl_module.local_rank == 0:
            metadata = {'dataloader_idx': dataloader_idx, 'sup_ids': sup_ids}
            fig, wav_reconstruction, wav_prediction, basename = synth_one_sample_with_target(
                _batch, output, self.vocoder, self.preprocess_config
            )
            figure_name = f"Validation/step_{pl_module.global_step}_{basename}"
            self.log_figure(logger, figure_name, fig, pl_module.global_step)
            self.log_audio(logger, "Validation", pl_module.global_step, basename, "reconstructed", wav_reconstruction, metadata)
            self.log_audio(logger, "Validation", pl_module.global_step, basename, "synthesized", wav_prediction, metadata)
            plt.close(fig)

    def _on_val_batch_end_for_validate(self, trainer, pl_module, outputs,
                                       batch, batch_idx, dataloader_idx):
        global_step = getattr(pl_module, 'test_global_step', pl_module.global_step)
        adaptation_steps = pl_module.adaptation_steps           # log/save period
        test_adaptation_steps = pl_module.test_adaptation_steps # total fine-tune steps
        sup_ids = batch[0][0][0][0]
        qry_ids = batch[0][1][0][0]
        SQids = f"{'-'.join(sup_ids)}.{'-'.join(qry_ids)}"
        task_id = trainer.datamodule.val_SQids2Tid[SQids]
        x = []
        y = {
            "Total Loss": [],
            "Mel Loss": [],
            "Mel-Postnet Loss": [],
            "Pitch Loss": [],
            "Energy Loss": [],
            "Duration Loss": [],
        }
        for ft_step in range(test_adaptation_steps+1):
            if f"step_{ft_step}" in outputs:
                x.append(ft_step)
                loss_dict = loss2dict(outputs[f"step_{ft_step}"]["recon"]["losses"])
                for k in y:
                    y[k].append(loss_dict[k])
        fig, axs = plt.subplots(nrows=2, ncols=3, constrained_layout=True)
        for ax, k in zip(axs.flat, y):
            ax.set_title(k)
            ax.plot(x, y[k], 'o', ls='-', ms=4)
        figure_name = f"Validation/{task_id}_losses"
        figure_path = os.path.join(self.result_dir, f"figure/step-{global_step}",
                                   f"{figure_name}.png")
        os.makedirs(os.path.dirname(figure_path), exist_ok=True)
        # plt.savefig(figure_path)
        self.log_figure(pl_module.logger, figure_name, fig, global_step)
        plt.close()

    def on_test_batch_end(self, trainer, pl_module, all_outputs, batch, batch_idx, dataloader_idx):
        global_step = getattr(pl_module, 'test_global_step', pl_module.global_step)
        adaptation_steps = pl_module.adaptation_steps           # log/save period
        test_adaptation_steps = pl_module.test_adaptation_steps # total fine-tune steps
        self.vocoder.to(pl_module.device)

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
            os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

            loss_dicts = []
            _batch = outputs["_batch"]

            for ft_step in range(0, test_adaptation_steps+1, adaptation_steps):
                if ft_step == 0:
                    predictions = outputs[f"step_{ft_step}"]["recon"]["output"]
                    recon_samples(
                        _batch, predictions, self.vocoder, self.preprocess_config,
                        figure_dir, audio_dir
                    )

                if "recon" in outputs[f"step_{ft_step}"]:
                    valid_error = outputs[f"step_{ft_step}"]["recon"]["losses"]
                    loss_dicts.append({"Step": ft_step, **loss2dict(valid_error)})

                if "synth" in outputs[f"step_{ft_step}"]:
                    predictions = outputs[f"step_{ft_step}"]["synth"]["output"]
                    synth_samples(
                        _batch, predictions, self.vocoder, self.preprocess_config,
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
        wav_file_path = os.path.join(save_dir, filename)

        os.makedirs(os.path.dirname(wav_file_path), exist_ok=True)
        wavfile.write(wav_file_path, sample_rate, audio)

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
        csv_file_path = os.path.join(log_dir, f"{basename}.csv")
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

        df = pd.DataFrame(loss_dict, columns=CSV_COLUMNS, index=[step])
        df.to_csv(csv_file_path, mode='a', header=not os.path.exists(csv_file_path), index=True, index_label="Step")

    def log_figure(self, logger, figure_name, figure, step):
        """
        Parameters:
            logger (LightningLoggerBase): {pl.loggers.CometLogger, pl.loggers.TensorBoardLogger}.
            stage (str): {"Training", "Validation", "Testing"}.
            step (int): Current step.
            basename (str): Audio index (original filename).
            tag (str): {"reconstructed", "synthesized"}.
            figure (matplotlib.pyplot.figure):
        """
        # print(figure_name)
        # print(step)

        def _log_figure(_logger):
            if isinstance(_logger, pl.loggers.CometLogger):
                _logger.experiment.log_figure(
                    figure_name=figure_name,
                    figure=figure,
                    step=step,
                )
            elif isinstance(_logger, pl.loggers.TensorBoardLogger):
                _logger.experiment.add_figure(
                    tag=figure_name,
                    figure=figure,
                    global_step=step,
                )
            else:
                print("Failed to log figure: not finding correct logger type")

        if isinstance(logger, LoggerCollection):
            for _logger in logger:
                _log_figure(_logger)
        else:
            _log_figure(logger)

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

        def _log_audio(_logger):
            if isinstance(_logger, pl.loggers.CometLogger):
                if metadata is None:
                    metadata = {}
                metadata.update({'stage': stage})
                _logger.experiment.log_audio(
                    audio_data=audio / max(abs(audio)),
                    sample_rate=sample_rate,
                    file_name=f"{basename}_{tag}.wav",
                    step=step,
                    metadata=metadata,
                )
            elif isinstance(_logger, pl.loggers.TensorBoardLogger):
                _logger.experiment.add_audio(
                    tag=f"{stage}/step_{step}_{basename}_{tag}",
                    snd_tensor=audio / max(abs(audio)),
                    global_step=step,
                    sample_rate=sample_rate,
                )
            else:
                print("Failed to log audio: not finding correct logger type")

        if isinstance(logger, LoggerCollection):
            for _logger in logger:
                _log_audio(_logger)
        else:
            _log_audio(logger)
