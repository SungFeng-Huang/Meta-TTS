import torch

from .system import System


class AdaptorSystem(System):
    """ Abstract class for few shot adaptation test. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # All of the settings below are for few-shot validation
        self.adaptation_lr = self.algorithm_config["adapt"]["task"]["lr"]
        self.adaptation_class = self.algorithm_config["adapt"]["class"]
        self.adaptation_steps = self.algorithm_config["adapt"]["train"]["steps"]
        self.test_adaptation_steps = self.algorithm_config["adapt"]["test"]["steps"]
        assert self.adaptation_steps == 0 or self.test_adaptation_steps % self.adaptation_steps == 0
    
    def _on_meta_batch_start(self, batch):
        """ Check meta-batch data """
        assert len(batch) == 1, "meta_batch_per_gpu"
        assert len(batch[0]) == 2 or len(batch[0]) == 4, "sup + qry (+ ref_phn_feats + lang_id)"
        assert len(batch[0][0]) == 1, "n_batch == 1"
        assert len(batch[0][0][0]) == 12, "data with 12 elements"

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_meta_batch_start(batch)
    
    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_meta_batch_start(batch)
    
    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_meta_batch_start(batch)

    @torch.enable_grad()
    def test_step(self, batch, batch_idx):
        outputs = {}
        for test_name, test_fn in getattr(self, "test_list", {}).items(): 
            outputs[test_name] = test_fn(batch, batch_idx)

        return outputs
       