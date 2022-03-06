#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
from tqdm import tqdm
import copy

from ..utils import MAML, CodebookAnalyzer
from utils.tools import get_mask_from_lengths
from ..adaptor import AdaptorSystem
from lightning.systems.utils import Task
from lightning.utils import loss2dict, LightningMelGAN
from lightning.model.phoneme_embedding import PhonemeEmbedding
from lightning.model import FastSpeech2Loss, FastSpeech2
from lightning.callbacks import Saver
from lightning.callbacks.utils import synth_samples, recon_samples


STATSDICT = {
    0: "./preprocessed_data/miniLibriTTS",
    1: "./preprocessed_data/miniAISHELL-3",
    2: "./preprocessed_data/miniGlobalPhone-fr",
    3: "./preprocessed_data/miniGlobalPhone-de",
    4: "",
    5: "./preprocessed_data/miniGlobalPhone-es",
    6: "./preprocessed_data/miniJVS",
    7: "./preprocessed_data/miniGlobalPhone-cz",
}

class MetaSystem(AdaptorSystem):
    """ 
    Concrete class with ANIL fastspeech2.
    Support:
        MAML Baseline
        HardAttCodeBook
        SoftAttCodeBook
        SoftMultiAttCodeBook
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_codebook_type()

        # tests
        self.codebook_analyzer = CodebookAnalyzer(self.result_dir)
        self.test_list = {
            "codebook visualization": self.visualize_matching,
            "adaptation": self.test_adaptation, 
        }

    def build_model(self):
        self.embedding_model = PhonemeEmbedding(self.model_config, self.algorithm_config)
        self.model = FastSpeech2(self.preprocess_config, self.model_config, self.algorithm_config)
        self.loss_func = FastSpeech2Loss(self.preprocess_config, self.model_config)

        self.vocoder = LightningMelGAN()
        self.vocoder.freeze()

    def build_optimized_model(self):
        return nn.ModuleList([self.embedding_model, self.model])

    def build_saver(self):
        saver = Saver(self.preprocess_config, self.log_dir, self.result_dir)
        saver.set_meta_saver()
        return saver
    
    def init_codebook_type(self):        
        codebook_config = self.algorithm_config["adapt"]["phoneme_emb"]
        if codebook_config["type"] == "embedding":
            self.codebook_type = "table-sep"
        elif codebook_config["type"] == "codebook":
            self.codebook_type = codebook_config["attention"]["type"]
        else:
            raise NotImplementedError
        self.adaptation_steps = self.algorithm_config["adapt"]["train"]["steps"]

    def build_learner(self, batch):
        _, _, ref_phn_feats, lang_id = batch[0]
        embedding = self.embedding_model.get_new_embedding(self.codebook_type, ref_phn_feats=ref_phn_feats, lang_id=lang_id)
        
        # Directly assign weight here, do not turn embedding into nn.Parameters since gradients can not flow back!
        # e.g. do not use emb_layer.weights = ... or nn.Embedding.from_pretrained
        emb_layer = nn.Embedding(*embedding.shape, padding_idx=0).to(self.device)
        emb_layer._parameters['weight'] = embedding
        adapt_dict = nn.ModuleDict({
            k: getattr(self.model, k) for k in self.algorithm_config["adapt"]["modules"]
        })
        adapt_dict["embedding"] = emb_layer
        return MAML(adapt_dict, lr=self.adaptation_lr)
    
    def forward_learner(
        self, learner, speaker_args, texts, src_lens, max_src_len,
        mels=None, mel_lens=None, max_mel_len=None,
        p_targets=None, e_targets=None, d_targets=None,
        p_control=1.0, e_control=1.0, d_control=1.0,
        average_spk_emb=False,
    ):
        _get_module = lambda name: getattr(learner.module, name, getattr(self.model, name, None))
        embedding        = _get_module('embedding')
        encoder          = _get_module('encoder')
        variance_adaptor = _get_module('variance_adaptor')
        decoder          = _get_module('decoder')
        mel_linear       = _get_module('mel_linear')
        postnet          = _get_module('postnet')
        speaker_emb      = _get_module('speaker_emb')

        if p_targets is not None:
            p_targets = p_targets.contiguous()
            e_targets = e_targets.contiguous()

        src_masks = get_mask_from_lengths(src_lens, max_src_len)

        # print("text ids", texts)
        emb_texts = embedding(texts)
        if (emb_texts != emb_texts).any():
            print("NaN table")
        output = encoder(emb_texts, src_masks)
        if (output != output).any():
            print("encoder nan")

        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None
        )

        if speaker_emb is not None:
            spk_emb = speaker_emb(speaker_args)
            if average_spk_emb:
                spk_emb = spk_emb.mean(dim=0, keepdim=True).expand(output.shape[0], -1)
            output += spk_emb.unsqueeze(1).expand(-1, max_src_len, -1)

        (
            output, p_predictions, e_predictions, log_d_predictions, d_rounded,
            mel_lens, mel_masks,
        ) = variance_adaptor(
            output, src_masks, mel_masks, max_mel_len,
            p_targets, e_targets, d_targets, p_control, e_control, d_control,
        )
        if (output != output).any():
            print("variance_adaptor nan")

        if speaker_emb is not None:
            spk_emb = speaker_emb(speaker_args)
            if average_spk_emb:
                spk_emb = spk_emb.mean(dim=0, keepdim=True).expand(output.shape[0], -1)
            if max_mel_len is None:  # inference stage
                max_mel_len = max(mel_lens)
            output += spk_emb.unsqueeze(1).expand(-1, max_mel_len, -1)

        output, mel_masks = decoder(output, mel_masks)
        if (output != output).any():
            print("decoder nan")
        output = mel_linear(output)
        if (output != output).any():
            print("mel linear nan")

        tmp = postnet(output)
        if (tmp != tmp).any():
            print("postnet nan")
        postnet_output = tmp + output

        return (
            output, postnet_output,
            p_predictions, e_predictions, log_d_predictions, d_rounded,
            src_masks, mel_masks, src_lens, mel_lens,
        )
    
    # Second order gradients for RNNs
    @torch.backends.cudnn.flags(enabled=False)
    @torch.enable_grad()
    def adapt(self, batch, adaptation_steps=5, learner=None, task=None, train=True):
        # MAML
        first_order = not train
        for step in range(adaptation_steps):
            mini_batch = task.next_batch()

            preds = self.forward_learner(learner, *mini_batch[2:])
            train_error = self.loss_func(mini_batch, preds)
            learner.adapt_(
                train_error[0], first_order=first_order,
                allow_unused=False, allow_nograd=True
            )
            print("train error", train_error)
            if train_error[0].isnan().any():
                assert 1 == 2
        return learner

    def meta_learn(self, batch, batch_idx, train=True):
        learner = self.build_learner(batch)
        sup_batch, qry_batch, _, _ = batch[0]
        batch = [(sup_batch, qry_batch)]
        sup_batch = batch[0][0][0]
        qry_batch = batch[0][1][0]

        if self.adaptation_steps > 0:
            task = Task(sup_data=sup_batch,
                        qry_data=qry_batch,
                        batch_size=self.algorithm_config["adapt"]["imaml"]["batch_size"]) # The batch size is borrowed from the imaml config.
            
            learner = learner.clone()
            learner = self.adapt(batch, min(self.adaptation_steps, self.test_adaptation_steps), learner=learner, task=task, train=train)

        # Evaluating the adapted model
        predictions = self.forward_learner(learner, *qry_batch[2:], average_spk_emb=True)
        valid_error = self.loss_func(qry_batch, predictions)
        return valid_error, predictions

    def training_step(self, batch, batch_idx):
        train_loss, predictions = self.meta_learn(batch, batch_idx, train=True)
        qry_batch = batch[0][1][0]

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v for k, v in loss2dict(train_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': train_loss[0], 'losses': train_loss, 'output': predictions, '_batch': qry_batch}

    def validation_step(self, batch, batch_idx):
        val_loss, predictions = self.meta_learn(batch, batch_idx)
        qry_batch = batch[0][1][0]

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v for k, v in loss2dict(val_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'losses': val_loss, 'output': predictions, '_batch': qry_batch}

    def test_step(self, batch, batch_idx):
        return super().test_step(batch, batch_idx)["adaptation"]
    
    def visualize_matching(self, batch, batch_idx):
        if self.codebook_type != "table-sep":
            _, _, ref_phn_feats, lang_id = batch[0]
            matching = self.embedding_model.get_matching(self.codebook_type, ref_phn_feats=ref_phn_feats, lang_id=lang_id)
            self.codebook_analyzer.visualize_matching(batch_idx, matching)
        return None

    def test_adaptation(self, batch, batch_idx):
        learner = self.build_learner(batch)
        outputs = {}

        sup_batch, qry_batch, _, lang_id = batch[0]
        batch = [(sup_batch, qry_batch)]
        sup_batch = batch[0][0][0]
        qry_batch = batch[0][1][0]

        # Create result directory
        task_id = f"test_{batch_idx:03d}"
        figure_dir = os.path.join(self.result_dir, "figure", "Testing", f"step_{self.test_global_step}", task_id)
        audio_dir = os.path.join(self.result_dir, "audio", "Testing", f"step_{self.test_global_step}", task_id)
        figure_fit_dir = os.path.join(self.result_dir, "figure-fit", "Testing", f"step_{self.test_global_step}", task_id)
        audio_fit_dir = os.path.join(self.result_dir, "audio-fit", "Testing", f"step_{self.test_global_step}", task_id)
        os.makedirs(figure_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(figure_fit_dir, exist_ok=True)
        os.makedirs(audio_fit_dir, exist_ok=True)
        config = copy.deepcopy(self.preprocess_config)
        config["path"]["preprocessed_path"] = STATSDICT[lang_id]

        # Build mini-batches
        task = Task(sup_data=sup_batch,
                    qry_data=qry_batch,
                    batch_size=self.algorithm_config["adapt"]["test"]["batch_size"])

        # Evaluate some training data to check overfit.
        fit_batch = task.next_batch()
        outputs['_batch_fit'] = fit_batch
        outputs['_batch'] = qry_batch

        # Evaluating the initial model (zero shot)
        learner.eval()
        self.model.eval()
        with torch.no_grad():
            fit_preds = self.forward_learner(learner, *fit_batch[2:], average_spk_emb=True)
            # outputs["step_0"] = {"recon-fit": {"output": fit_preds}}

            predictions = self.forward_learner(learner, *qry_batch[2:], average_spk_emb=True)
            valid_error = self.loss_func(qry_batch, predictions)
            outputs["step_0"] = {"recon": {"losses": valid_error}}
      
            # synth_samples & save & log
            # No reference from unseen speaker, use reference from support set instead.
            recon_samples(
                fit_batch, fit_preds, self.vocoder, config,
                figure_fit_dir, audio_fit_dir
            )
            recon_samples(
                qry_batch, predictions, self.vocoder, config,
                figure_dir, audio_dir
            )

            predictions = self.forward_learner(learner, sup_batch[2], *qry_batch[3:6], average_spk_emb=True)
            synth_samples(
                fit_batch, fit_preds, self.vocoder, config,
                figure_fit_dir, audio_fit_dir, f"step_{self.test_global_step}-FTstep_0"
            )
            synth_samples(
                qry_batch, predictions, self.vocoder, config,
                figure_dir, audio_dir, f"step_{self.test_global_step}-FTstep_0"
            )
        learner.train()
        self.model.train()

        # Determine fine tune checkpoints.
        ft_steps = list(range(5, 201, 5))
        if batch_idx == 0:
            return outputs
        
        # Adapt
        learner = learner.clone()
        self.test_adaptation_steps = max(ft_steps)
        
        for ft_step in tqdm(range(self.adaptation_steps, self.test_adaptation_steps+1, self.adaptation_steps)):
            learner = self.adapt(batch, self.adaptation_steps, learner=learner, task=task, train=False)
            
            learner.eval()
            self.model.eval()
            with torch.no_grad():
                # Evaluating the adapted model
                predictions = self.forward_learner(learner, *qry_batch[2:], average_spk_emb=True)
                valid_error = self.loss_func(qry_batch, predictions)
                outputs[f"step_{ft_step}"] = {"recon": {"losses": valid_error}}

                # synth_samples & save & log
                if ft_step in ft_steps:
                    fit_preds = self.forward_learner(learner, *fit_batch[2:], average_spk_emb=True)
                    # outputs[f"step_{ft_step}"].update({"synth-fit": {"output": fit_preds}})
                    predictions = self.forward_learner(learner, sup_batch[2], *qry_batch[3:6], average_spk_emb=True)
                    # outputs[f"step_{ft_step}"].update({"synth": {"output": predictions}})

                    synth_samples(
                        fit_batch, fit_preds, self.vocoder, config,
                        figure_fit_dir, audio_fit_dir, f"step_{self.test_global_step}-FTstep_{ft_step}"
                    )
                    synth_samples(
                        qry_batch, predictions, self.vocoder, config,
                        figure_dir, audio_dir, f"step_{self.test_global_step}-FTstep_{ft_step}"
                    )
            learner.train()
            self.model.train()
        del learner

        return outputs
