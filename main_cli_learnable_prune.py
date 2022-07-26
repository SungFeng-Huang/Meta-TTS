import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI


if __name__ == "__main__":
    from lightning.systems.prune.learnable_prune import LearnablePruneSystem
    from lightning.datamodules.learnable_prune_datamodule import LearnablePruneDataModule

    cli = LightningCLI(
        LearnablePruneSystem, LearnablePruneDataModule,
        subclass_mode_model=True, subclass_mode_data=True)

