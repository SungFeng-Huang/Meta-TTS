###
# Usage: python main_cli.py [-c config_with_subcommand] subcommand
# Usage: python main_cli.py subcommand [-c config]
###
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI


if __name__ == "__main__":
    from lightning.systems.base_adapt import BaseAdaptorSystem
    from lightning.datamodules.base_datamodule import BaseDataModule

    cli = LightningCLI(
        BaseAdaptorSystem, BaseDataModule,
        subclass_mode_model=True, subclass_mode_data=True)

