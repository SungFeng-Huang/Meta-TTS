###
# Usage: python main_cli.py [-c config_with_subcommand] subcommand [-c config]
###
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # "default_config_files" cannot use with argument linking.
        # AttributeError: 'NoneType' object has no attribute 'keys'
        # jsonargparse.util.ParserError: Problem in default config file ::
        #    argument of type 'NoneType' is not iterable
        # parser.link_arguments("data.num_classes",
        #                       "model.init_args.model.init_args.num_classes",
        #                       apply_on="instantiate")
        parser.link_arguments("data.num_classes",
                              "model.init_args.num_classes",
                              apply_on="instantiate")


if __name__ == "__main__":
    from projects.xvec.model import XvecTDNN
    from lightning.datamodules import XvecDataModule

    cli = MyLightningCLI(
        XvecTDNN, XvecDataModule,
        parser_kwargs={
            "fit": {
                "default_config_files": ["cli_config/xvec/fit.accent.yaml"],
            },
            "test": {
                "default_config_files": ["cli_config/xvec/test.yaml"],
            },
        },
        subclass_mode_model=True, subclass_mode_data=True)
