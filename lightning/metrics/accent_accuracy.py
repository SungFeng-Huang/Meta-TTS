from typing import Union, Optional, Dict, Any

from torch import Tensor
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cli import instantiate_class
from torchmetrics import Accuracy

from lightning.model import XvecTDNN


class AccentAccuracy(Accuracy, LightningModule):
    def __init__(self,
                 model: Optional[Union[XvecTDNN, Dict]] = None,
                 model_dict: Optional[Dict] = None,
                 num_classes: Optional[int] = None,
                 **kwargs: Dict[str, Any]):
        """Accent classification accuracy.

        Args:
            model (optional): X-vector model. If specified, `model_dict` would
                be ignored.
            model_dict (optional): X-vector model config. If 'model' is not
                specified, would use `model_dict` instead.
        """
        super().__init__(**kwargs)

        if model is None:
            if model_dict is None:
                raise TypeError
            model = model_dict

        if isinstance(model, Dict):
            num_classes = model["init_args"].get("num_classes", num_classes)
            if num_classes is None:
                raise ValueError
            model["init_args"]["num_classes"] = num_classes
            self.model = instantiate_class((), model)
        elif isinstance(model, XvecTDNN):
            self.model = model
        else:
            raise TypeError

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
