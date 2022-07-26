import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI


if __name__ == "__main__":
    from lightning.systems.prune.prune_accent import PruneAccentSystem
    from lightning.datamodules.prune_accent_datamodule import PruneAccentDataModule

    cli = LightningCLI(
        PruneAccentSystem, PruneAccentDataModule,
        subclass_mode_model=True, subclass_mode_data=True)

