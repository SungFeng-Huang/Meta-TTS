from os import PathLike
from typing import Dict, Any, Union

from torch import Tensor
from torchmetrics import Accuracy

from lightning.model import XvecTDNN


class XvecAccentAccuracy(Accuracy):
    def __init__(self,
                 ckpt_path: Union[str, bytes, PathLike],
                 **kwargs: Dict[str, Any]):
        """Accent classification accuracy.

        Args:
            ckpt_path: Checkpoint path of pretrained XvecTDNN.
            kwargs: Accuracy kwargs.
        """
        super().__init__(**kwargs)

        self.model = XvecTDNN.load_from_checkpoint(ckpt_path)
        self.model.freeze()

    def update(self, mel_spec: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            mel_spec: Predicted or real Mel-Spectrograms.
            target: Ground truth labels.
        """
        if mel_spec.squeeze.ndim > 3:
            raise ValueError(
                "Number of dimension too large: "
                f"expected 2 or 3, get {mel_spec.squeeze.ndim}")
        if mel_spec.shape[1] != self.model.tdnn1.in_channels:
            if mel_spec.shape[2] != self.model.tdnn1.in_channels:
                raise ValueError(
                    "Dimension mismatch between mel_spec and XvecTDNN.")
            mel_spec = mel_spec.transpose(1, 2)

        self.model.eval()

        preds = self.model(mel_spec)

        super().update(preds, target)
