from abc import ABC
import pytorch_lightning as pl


class FSCLPlugIn(pl.LightningModule):
    """ Interface for FSCLPlugIn """    
    def build_optimized_model(self, *args, **kwargs):
        raise NotImplementedError
    
    def build_embedding_table(self, ref_infos, return_attn=False, *args, **kwargs):
        raise NotImplementedError
    
    # def build_segmental_representation(self, ref_infos, *args, **kwargs):
    #     raise NotImplementedError

    def on_save_checkpoint(self, checkpoint, prefix="fscl", *args, **kwargs):
        raise NotImplementedError


class Hookable(ABC):
    def build_hook(self, key: str, *args, **kwargs):
        raise NotImplementedError
    
    def get_hook(self, key: str, *args, **kwargs):
        raise NotImplementedError


class Tunable(ABC):
    def tune_init(self, data_configs, *args, **kwargs) -> None:
        raise NotImplementedError
