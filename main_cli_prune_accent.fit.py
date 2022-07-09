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


if __name__ == "__main__":
    from lightning.systems.base_adapt.prune_accent import PruneAccentSystem
    from lightning.datamodules.prune_accent_datamodule import PruneAccentDataModule

    cli = LightningCLI(
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

