#!/usr/bin/env python3
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, GPUStatsMonitor, ModelCheckpoint

from lightning.callbacks import GlobalProgressBar
from lightning.optimizer import get_optimizer
from lightning.scheduler import get_scheduler


class System(pl.LightningModule):
    """ Abstract base class for all systems. """

    default_monitor: str = "val_loss"

    def __init__(self, preprocess_config, model_config, train_config, algorithm_config, log_dir, result_dir):
        super().__init__()
        self.preprocess_config = preprocess_config
        self.model_config = model_config
        self.train_config = train_config
        self.algorithm_config = algorithm_config
        self.save_hyperparameters()

        self.log_dir = log_dir
        self.result_dir = result_dir
        
        self.build_model()

    def build_model(self):
        """ Build all components here. """
        raise NotImplementedError
    
    def build_optimized_model(self):
        """ Return modules to be updated in training loop, if meta learning, consider only outer loop. """
        raise NotImplementedError    

    def build_saver(self):
        """ Return a saver class. """
        # return Saver(self.preprocess_config, self.log_dir, self.result_dir)
        raise NotImplementedError
    
    def configure_callbacks(self):
        # Checkpoint saver
        save_step = self.train_config["step"]["save_step"]
        checkpoint = ModelCheckpoint(
            monitor="Val/Total Loss", mode="min",
            every_n_train_steps=save_step, save_top_k=-1, save_last=True,
        )

        # Progress bars (step/epoch)
        outer_bar = GlobalProgressBar(process_position=1)

        # Monitor learning rate / gpu stats
        lr_monitor = LearningRateMonitor()
        gpu_monitor = GPUStatsMonitor(
            memory_utilization=True, gpu_utilization=True, intra_step_time=True, inter_step_time=True
        )
        
        # Save figures/audios/csvs
        saver = self.build_saver()
        if isinstance(saver, list):
            callbacks = [checkpoint, outer_bar, lr_monitor, gpu_monitor, *saver]
        else:
            callbacks = [checkpoint, outer_bar, lr_monitor, gpu_monitor, saver]
        return callbacks

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        self.optimizer = get_optimizer(self.build_optimized_model(), self.model_config, self.train_config)

        self.scheduler = {
            "scheduler": get_scheduler(self.optimizer, self.train_config),
            'interval': 'step', # "epoch" or "step"
            'frequency': 1,
            'monitor': self.default_monitor,
        }

        return [self.optimizer], [self.scheduler]

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        return checkpoint

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.test_global_step = checkpoint["global_step"]
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        state_dict_pop_keys = []
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    if self.local_rank == 0:
                        print(f"Skip loading parameter: {k}, "
                                    f"required shape: {model_state_dict[k].shape}, "
                                    f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                if self.local_rank == 0:
                    print(f"Dropping parameter {k}")
                state_dict_pop_keys.append(k)
                is_changed = True

        # modify state_dict format to model_state_dict format
        for k in state_dict_pop_keys:
            state_dict.pop(k)
        for k in model_state_dict:
            if k not in state_dict:
                state_dict[k] = model_state_dict[k]

        if is_changed:
            checkpoint.pop("optimizer_states", None)
