from .asr_saver import Saver as ASRSaver
from .saver import Saver as TTSSaver
from pytorch_lightning.callbacks import Callback


class Saver(Callback):

    def __init__(self, preprocess_config, log_dir=None, result_dir=None):
        super().__init__()
        self.asr_saver = ASRSaver(preprocess_config, log_dir=log_dir, result_dir=result_dir)
        self.tts_saver = TTSSaver(preprocess_config, log_dir=log_dir, result_dir=result_dir)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        outputs["losses"] = outputs['dual_losses'][0]
        outputs['output'] = outputs['dual_output'][0]
        self.tts_saver.on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        outputs["losses"] = outputs['dual_losses'][1]
        outputs['output'] = outputs['dual_output'][1]
        self.asr_saver.on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.tts_saver.on_validation_epoch_start(trainer, pl_module)
        self.asr_saver.on_validation_epoch_start(trainer, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        outputs["losses"] = outputs['dual_losses'][0]
        outputs['output'] = outputs['dual_output'][0]
        self.tts_saver.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        outputs["losses"] = outputs['dual_losses'][1]
        outputs['output'] = outputs['dual_output'][1]
        self.asr_saver.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.tts_saver.on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)