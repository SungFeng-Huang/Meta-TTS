import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
from typing import Any, Callable


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.init_args.mask_steps_per_epoch",
                              "model.init_args.mask_steps_per_epoch",
                              apply_on="instantiate")
        parser.link_arguments("data.init_args.libri_mask",
                              "model.init_args.libri_mask",
                              apply_on="instantiate")

def change_default(func: Callable, arg_name: str, value: Any):
    """hacky way to change default value of positional argument"""
    # get function arguments
    n_args = func.__code__.co_argcount  # counts
    arg_names = func.__code__.co_varnames[:n_args]  # names

    # get default arguments
    df_vals = list(func.__defaults__)   # values
    n_df_args = len(df_vals)    # counts
    df_names = arg_names[-n_df_args:]   # names

    # change argument default value
    df_vals[df_names.index(arg_name)] = value
    func.__defaults__ = tuple(df_vals)


if __name__ == "__main__":
    import sys
    import os
    current_dir =  os.path.abspath(os.path.dirname(__file__))
    root_dir = os.path.abspath(current_dir + "/../../")
    sys.path.insert(0, root_dir)

    import traceback
    from projects.prune.systems.prune_accent import PruneAccentSystem
    from projects.prune.datamodules.prune_accent_datamodule import PruneAccentDataModule

    # change dendrogram default for seaborn.clustermap
    from scipy.cluster.hierarchy import dendrogram
    change_default(dendrogram, "distance_sort", True)

    try:
        cli = MyLightningCLI(
            PruneAccentSystem, PruneAccentDataModule,
            subclass_mode_model=True, subclass_mode_data=True)
    except:
        traceback.print_exc(
            file=open(
                os.path.join(cli.trainer.log_dir, "fit", "traceback.log"),
                'w'
            )
        )

