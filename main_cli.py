import pytorch_lightning as pl

from pytorch_lightning.utilities.cli import LightningCLI, MODEL_REGISTRY, DATAMODULE_REGISTRY


# MODEL_REGISTRY.register_classes(pl.LightningModule)
# DATAMODULE_REGISTRY.register_classes(pl.LightningDataModule)

# class MyLightningCLI(LightningCLI):
#     def add_arguments_to_parser(self, parser):
#         parser.link_arguments("model.total_steps",
#                               "trainer.estimated_stepping_batches",
#                               apply_on="instantiate")

cli = LightningCLI(pl.LightningModule, pl.LightningDataModule,
                   subclass_mode_model=True, subclass_mode_data=True)
