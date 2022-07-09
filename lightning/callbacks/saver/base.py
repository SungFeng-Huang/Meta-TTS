import os
import pandas as pd
import pytorch_lightning as pl

from scipy.io import wavfile
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.base import LoggerCollection

from lightning.utils import LightningMelGAN


CSV_COLUMNS = ["Total Loss", "Mel Loss", "Mel-Postnet Loss", "Pitch Loss", "Energy Loss", "Duration Loss"]
COL_SPACE = [len(col) for col in ["200000", "Validation"]+CSV_COLUMNS]  # max step: 200000, longest stage: validation


class BaseSaver(Callback):
    """
    """

    def __init__(self, preprocess_config, log_dir=None, result_dir=None):
        super().__init__()
        self.preprocess_config = preprocess_config

        # self.log_dir = log_dir
        # self.result_dir = result_dir
        # os.makedirs(self.log_dir, exist_ok=True)
        # os.makedirs(self.result_dir, exist_ok=True)
        # print("Log directory:", self.log_dir)
        # print("Result directory:", self.result_dir)

        self.vocoder = LightningMelGAN()
        self.vocoder.freeze()
        self.vocoder.eval()

    def save_audio(self, stage, step, basename, tag, audio):
        """
        Args:
            stage (str): {"Training", "Validation", "Testing"}.
            step (int): Current step.
            basename (str): Audio index (original filename).
            tag (str): {"reconstructed", "synthesized"}.
            audio (numpy): Audio waveform.
        """
        sample_rate = self.preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        save_dir = os.path.join(self.log_dir, stage, "audio")
        filename = f"step_{step}_{basename}_{tag}.wav"
        wav_file_path = os.path.join(save_dir, filename)

        os.makedirs(os.path.dirname(wav_file_path), exist_ok=True)
        wavfile.write(wav_file_path, sample_rate, audio)

    def log_figure(self, logger, figure_name, figure, step):
        """
        Args:
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

    def log_audio(self, logger, stage, step, basename, tag, audio,
                  metadata=None, save=False):
        """
        Args:
            logger (LightningLoggerBase): {pl.loggers.CometLogger, pl.loggers.TensorBoardLogger}.
            stage (str): {"Training", "Validation", "Testing"}.
            step (int): Current step.
            basename (str): Audio index (original filename).
            tag (str): {"reconstructed", "synthesized"}.
            audio (numpy): Audio waveform.
        """
        sample_rate = self.preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        if save:
            self.save_audio(stage, step, basename, tag, audio)
        if metadata is None:
            metadata = {}
        metadata.update({'stage': stage})

        def _log_audio(_logger):
            if isinstance(_logger, pl.loggers.CometLogger):
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

    def _log_audio(self, logger, file_name, audio, step, sample_rate, *args, **kwargs):
        if isinstance(logger, LoggerCollection):
            for _logger in logger:
                self._log_audio(_logger, *args, **kwargs)
            return

        if isinstance(logger, pl.loggers.CometLogger):
            logger.experiment.log_audio(
                file_name=file_name,
                audio_data=audio / max(abs(audio)),
                sample_rate=sample_rate,
                step=step,
                *args, **kwargs)
        elif isinstance(logger, pl.loggers.TensorBoardLogger):
            logger.experiment.add_audio(
                tag=file_name,
                snd_tensor=audio / max(abs(audio)),
                sample_rate=sample_rate,
                global_step=step,
                *args, **kwargs)
        else:
            raise TypeError("Failed to log audio: not finding correct logger type")
