import os
import pandas as pd
from scipy.io import wavfile
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class BaseSaver(Callback):
    """
    Base class for all savers.
    """

    def __init__(self, log_dir=None, result_dir=None):
        super().__init__()
        self.log_dir = log_dir
        self.result_dir = result_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        print("Log directory:", self.log_dir)
        print("Result directory:", self.result_dir)

    def save_audio(self, stage, step, basename, tag, audio, sr):
        """
        Parameters:
            stage (str): {"Training", "Validation", "Testing"}.
            step (int): Current step.
            basename (str): Audio index (original filename).
            tag (str): {"reconstructed", "synthesized"}.
            audio (numpy): Audio waveform.
            sr (int): Audio sample rate.
        """
        save_dir = os.path.join(self.log_dir, "audio", stage)
        filename = f"step_{step}_{basename}_{tag}.wav"

        os.makedirs(save_dir, exist_ok=True)
        wavfile.write(os.path.join(save_dir, filename), sr, audio)

    def save_csv(self, stage, step, basename, loss_dict):
        if stage in ("Training", "Validation"):
            log_dir = os.path.join(self.log_dir, "csv", stage)
        else:
            log_dir = os.path.join(self.result_dir, "csv", stage)
        os.makedirs(log_dir, exist_ok=True)
        csv_file_path = os.path.join(log_dir, f"{basename}.csv")

        df = pd.DataFrame(loss_dict, columns=list(loss_dict.keys()), index=[step])
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

    def log_audio(self, logger, stage, step, basename, tag, audio, sr, metadata=None):
        """
        Parameters:
            logger (LightningLoggerBase): {pl.loggers.CometLogger, pl.loggers.TensorBoardLogger}.
            stage (str): {"Training", "Validation", "Testing"}.
            step (int): Current step.
            basename (str): Audio index (original filename).
            tag (str): {"reconstructed", "synthesized"}.
            audio (numpy): Audio waveform.
            sr (int): Audio sample rate.
        """
        self.save_audio(stage, step, basename, tag, audio, sr)

        if isinstance(logger, pl.loggers.CometLogger):
            if metadata is None:
                metadata = {}
            metadata.update({'stage': stage, 'type': tag, 'id': basename})
            logger.experiment.log_audio(
                audio_data=audio / max(abs(audio)),
                sample_rate=sr,
                file_name=f"{basename}_{tag}.wav",
                step=step,
                metadata=metadata,
            )
        elif isinstance(logger, pl.loggers.TensorBoardLogger):
            logger.experiment.add_audio(
                tag=f"{stage}/step_{step}_{basename}_{tag}",
                snd_tensor=audio / max(abs(audio)),
                global_step=step,
                sample_rate=sr,
            )
        else:
            print("Failed to log audio: not finding correct logger type")
