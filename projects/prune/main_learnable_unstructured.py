import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI


if __name__ == "__main__":
    import sys
    current_dir =  os.path.abspath(os.path.dirname(__file__))
    root_dir = os.path.abspath(current_dir + "/../../")
    sys.path.insert(0, root_dir)

    from projects.prune.systems.prune_accent import PruneAccentSystem
    from projects.prune.datamodules.prune_accent_datamodule import PruneAccentDataModule

    cli = LightningCLI(
        PruneAccentSystem, PruneAccentDataModule,
        subclass_mode_model=True, subclass_mode_data=True)

