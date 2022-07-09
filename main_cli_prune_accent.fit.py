from pytorch_lightning.callbacks.pruning import ModelPruning
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.loops.epoch.training_epoch_loop import \
    TrainingEpochLoop, _OUTPUTS_TYPE

class MyEpochLoop(TrainingEpochLoop):
    """Also validate on train epoch start."""
    def on_run_start(self, data_fetcher) -> None:
        super().on_run_start(data_fetcher)
        self.trainer.validating = True
        self._run_validation()
        self.trainer.training = True
    # def on_run_end(self) -> _OUTPUTS_TYPE:
    #     self.trainer.validating = True
    #     self._run_validation()
    #     self.trainer.training = True
    #     outputs, self._outputs = self._outputs, []
    #     return outputs

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(ModelPruning, "model_pruning_callback")
        parser.set_defaults({
            "model_pruning_callback.pruning_fn": "l1_unstructured",
            "model_pruning_callback.parameters_to_prune": None,
            "model_pruning_callback.use_global_unstructured": True,
            "model_pruning_callback.amount": 0.1,
            "model_pruning_callback.apply_pruning": True,
            "model_pruning_callback.use_lottery_ticket_hypothesis": True,
            "model_pruning_callback.resample_parameters": False,
            "model_pruning_callback.verbose": 2,
            "model_pruning_callback.prune_on_train_epoch_end": True,
        })


if __name__ == "__main__":
    from lightning.systems.base_adapt.prune_accent import PruneAccentSystem
    from lightning.datamodules.prune_accent_datamodule import PruneAccentDataModule

    cli = MyLightningCLI(
        PruneAccentSystem, PruneAccentDataModule,
        subclass_mode_model=True, subclass_mode_data=True,
        run=False,
    )

    # Optional: stitch back the trainer arguments
    epoch_loop = MyEpochLoop(cli.trainer.fit_loop.epoch_loop.min_steps,
                             cli.trainer.fit_loop.epoch_loop.max_steps)
# Optional: connect children loops as they might have existing state
    epoch_loop.connect(cli.trainer.fit_loop.epoch_loop.batch_loop,
                       cli.trainer.fit_loop.epoch_loop.val_loop)
# Instantiate and connect the loop.
    cli.trainer.fit_loop.connect(epoch_loop=epoch_loop)
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

