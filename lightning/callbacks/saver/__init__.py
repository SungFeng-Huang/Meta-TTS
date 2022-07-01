from .fit import FitSaver
from .validate import ValidateSaver
from .test import TestSaver


class Saver(FitSaver, ValidateSaver, TestSaver):
    def on_validation_start(self, trainer, pl_module):
        if trainer.state.fn == "fit":
            pass
        elif trainer.state.fn == "validate":
            return ValidateSaver.on_validation_start(self, trainer, pl_module)

    def on_validation_batch_end(self, trainer, pl_module,
                                outputs, batch, batch_idx, dataloader_idx):
        if trainer.state.fn == "fit":
            return FitSaver.on_validation_batch_end(
                self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        elif trainer.state.fn == "validate":
            return ValidateSaver.on_validation_batch_end(
                self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
