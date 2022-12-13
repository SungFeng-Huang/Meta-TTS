###
# Usage: python main_cli.py [-c config_with_subcommand] subcommand [-c config]
###
import os
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
    import sys
    current_dir =  os.path.abspath(os.path.dirname(__file__))
    root_dir = os.path.abspath(current_dir + "/../../")
    sys.path.insert(0, root_dir)

    from projects.xvec.model import XvecTDNN
    from projects.xvec.datamodule import XvecDataModule

    cli = MyLightningCLI(
        XvecTDNN, XvecDataModule,
        parser_kwargs={
            "fit": {
                "default_config_files": ["cli_config/xvec/fit.accent.yaml"],
            },
            "test": {
                "default_config_files": ["cli_config/xvec/test.accent.yaml"],
            },
        },
        subclass_mode_model=True, subclass_mode_data=True)