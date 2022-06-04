import pytorch_lightning as pl

from pytorch_lightning.utilities.cli import LightningCLI, OPTIMIZER_REGISTRY, LR_SCHEDULER_REGISTRY


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args(
            OPTIMIZER_REGISTRY.classes, nested_key="optimizer",
            link_to="model.init_args.optim_config"
        )
        parser.add_lr_scheduler_args(
            LR_SCHEDULER_REGISTRY.classes, nested_key="lr_scheduler",
            link_to="model.init_args.scheduler_config"
        )
        parser.link_arguments("data.num_tgt",
                              "model.init_args.model.init_args.numSpkrs",
                              apply_on="instantiate")
        parser.link_arguments("data.num_tgt",
                              "model.init_args.num_tgt",
                              apply_on="instantiate")

cli = MyLightningCLI(pl.LightningModule, pl.LightningDataModule,
                     # parser_kwargs={"default_config_files":
                     #                ["cli_config/xvec.yaml"]},
                     subclass_mode_model=True, subclass_mode_data=True)

# "default_config_files" cannot use with `add_optimizer_args` and `add_lr_scheduler_args`
# jsonargparse.util.ParserError: Problem in default config file :: argument of type 'NoneType' is not iterable
