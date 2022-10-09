import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.init_args.mask_steps_per_epoch",
                              "model.init_args.mask_steps_per_epoch",
                              apply_on="instantiate")
        parser.link_arguments("data.init_args.libri_mask",
                              "model.init_args.libri_mask",
                              apply_on="instantiate")

if __name__ == "__main__":
    from lightning.systems.prune.prune_accent import PruneAccentSystem
    from lightning.datamodules.prune_accent_datamodule import PruneAccentDataModule

    cli = MyLightningCLI(
        PruneAccentSystem, PruneAccentDataModule,
        subclass_mode_model=True, subclass_mode_data=True)

