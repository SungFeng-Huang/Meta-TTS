#!/usr/bin/env python3

import time
import torch

from lightning.systems.base_adapt.base import BaseAdaptorSystem
from lightning.callbacks.saver import TestSaver
from lightning.systems.utils import Task


VERBOSE = False

class BaseAdaptorTestSystem(BaseAdaptorSystem):
    """A PyTorch Lightning module for ANIL for FastSpeech2.
    """

    def __init__(self, preprocess_config, model_config, train_config, algorithm_config, log_dir, result_dir):
        super().__init__(preprocess_config, model_config, train_config, algorithm_config, log_dir, result_dir)

        assert "saving_steps" in self.algorithm_config["adapt"]["test"]
        self.saving_steps = self.algorithm_config["adapt"]["test"]["saving_steps"]

    def configure_callbacks(self):
        # Save figures/audios/csvs
        saver = TestSaver(self.preprocess_config, self.log_dir, self.result_dir)
        self.saver = saver

        callbacks = [saver]
        return callbacks

    def on_test_start(self):
        if self.algorithm_config["adapt"]["speaker_emb"] == "table":
            if self.local_rank == 0:
                print("Testing speaker emb")
                print("Before:")
                print(self.model.speaker_emb.model.weight[-39:])

            if (self.preprocess_config["dataset"] == "LibriTTS"
                    and self.algorithm_config["adapt"]["test"].get("avg_train_spk_emb", False)):

                with torch.no_grad():
                    self.model.speaker_emb.model.weight[-39:] = \
                        self.model.speaker_emb.model.weight[:247].mean(dim=0)

            if self.local_rank == 0:
                print("After:")
                print(self.model.speaker_emb.model.weight[-39:])
                print()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        all_outputs = []
        qry_batch = batch[0][1][0]
        if self.algorithm_config["adapt"]["test"].get("1-shot", False):
            task = Task(sup_data=batch[0][0][0], batch_size=1, shuffle=False)
            for sup_batch in iter(task):
                mini_batch = [([sup_batch], [qry_batch])]
                outputs = self._test_step(mini_batch, batch_idx, dataloader_idx)
                all_outputs.append(outputs)
        else:
            outputs = self._test_step(batch, batch_idx, dataloader_idx)
            all_outputs.append(outputs)
        torch.distributed.barrier()

        return all_outputs

    def _test_step(self, batch, batch_idx, dataloader_idx):
        adapt_steps = [step_ - _step for _step, step_ in
                       zip([0] + self.saving_steps, self.saving_steps)]
        # sup_ids = batch[0][0][0][0]
        # qry_ids = batch[0][1][0][0]
        sup_ids = batch[0][0][0]["ids"]
        qry_ids = batch[0][1][0]["ids"]
        SQids = f"{'-'.join(sup_ids)}.{'-'.join(qry_ids)}"
        task_id = self.trainer.datamodule.val_SQids2Tid[SQids]

        outputs = {
            "_batch": batch[0][1][0],   # qry_batch
            "task_id": task_id,
        }

        # Evaluating the initial model
        st = time.time()
        # learner = self.clone_learner(inference=True)
        learner, val_loss, recon_preds, synth_preds = self.meta_learn(
            batch, adapt_steps=0, train=False, synth=True,
            recon=True,
        )
        outputs[f"step_0"] = {
            "recon": {"losses": val_loss, "output": recon_preds},
            "synth": {"output": synth_preds},
        }
        self._print_mem_diff(st, "Adapt 0")

        st = time.time()
        torch.cuda.empty_cache()
        self._print_mem_diff(st, "Empty cache")

        # Adapt
        for ft_step, adapt_step in zip(self.saving_steps, adapt_steps):
            st = time.time()
            learner, val_loss, predictions = self.meta_learn(
                batch, adapt_step, learner=learner, train=False,
                synth=True, recon=False,
            )
            outputs[f"step_{ft_step}"] = {
                "recon": {"losses": val_loss},
                "synth": {"output": predictions},
            }
            self._print_mem_diff(st, f"Adapt ~{ft_step} ({adapt_step})")

            st = time.time()
            torch.cuda.empty_cache()
            self._print_mem_diff(st, "Empty cache")

        del learner

        return outputs

