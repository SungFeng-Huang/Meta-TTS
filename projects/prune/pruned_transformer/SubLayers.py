from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward


class PrunedLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.device == self.weight.device
        assert input.shape[-1] == self.in_features
        if self.in_features:
            return super().forward(input)
        else:
            output = torch.zeros(
                *input.shape[:-1], self.out_features,
                device=input.device,
            )
            if self.bias is not None:
                output = output + self.bias
            return output


class PrunedConv1d(nn.Conv1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.in_channels and self.out_channels:
            # d_in > 0, d_out > 0
            return super().forward(input)
        elif self.out_channels:
            # d_in == 0, d_out > 0
            # conv would ourput wrong size
            output = torch.zeros(
                input.shape[0], self.out_channels, input.shape[2],
                device=input.device,
            )
            if self.bias is not None:
                output = output + self.bias[None, :, None]
            return output
        else:
            # d_out == 0
            return torch.randn(
                input.shape[0], 0, input.shape[2],
                device=input.device,
            )


class PrunedLayerNorm(nn.LayerNorm):
    def __init__(self, d_orig: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_orig = d_orig

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[-len(self.normalized_shape):] == self.normalized_shape, f"{input.shape}, {self.normalized_shape}"
        d_new = self.normalized_shape[0]
        dims = tuple(range(-len(self.normalized_shape), 0))

        m_new = input.mean(dim=dims, keepdim=True)

        m_orig = m_new * d_new / self.d_orig
        v_orig = (input**2).sum(dim=dims, keepdim=True)/self.d_orig - m_orig**2
        return (input - m_orig) / (v_orig+self.eps).sqrt() * self.weight + self.bias

class PrunedMultiHeadAttention(MultiHeadAttention):
    """ Multi-Head Attention module """

    def __init__(
        self,
        n_head: int = None,
        d_model: int = None,
        d_k: int = None,
        d_v: int = None,
        dropout: float = 0.1,
        orig: MultiHeadAttention = None,
    ):
        if orig:
            # copy constructor
            n_head = orig.n_head
            d_k = orig.d_k
            d_v = orig.d_v
            d_model = orig.fc.out_features
            dropout = orig.dropout.p
            super().__init__(n_head, d_model, d_k, d_v, dropout)
        elif n_head and d_model and d_k and d_v and dropout:
            # normal constructor
            super().__init__(n_head, d_model, d_k, d_v, dropout)

    def prune(
        self,
        n: str,
        model_state_dict: dict[str, dict[str, torch.Tensor]],
    ):
        w_LN_mask = model_state_dict["mask"][f"model.{n}.layer_norm.weight_mask"]   # (d_model)
        b_LN_mask = model_state_dict["mask"][f"model.{n}.layer_norm.bias_mask"]   # (d_model)
        self.d_model = getattr(self, "d_model", None) or w_LN_mask.shape[0]
        assert b_LN_mask.shape == (self.d_model,)

        w_qs_mask = model_state_dict["mask"][f"model.{n}.w_qs.weight_mask"]
        b_qs_mask = model_state_dict["mask"][f"model.{n}.w_qs.bias_mask"]
        w_ks_mask = model_state_dict["mask"][f"model.{n}.w_ks.weight_mask"]
        b_ks_mask = model_state_dict["mask"][f"model.{n}.w_ks.bias_mask"]
        w_vs_mask = model_state_dict["mask"][f"model.{n}.w_vs.weight_mask"]
        b_vs_mask = model_state_dict["mask"][f"model.{n}.w_vs.bias_mask"]
        w_fc_mask = model_state_dict["mask"][f"model.{n}.fc.weight_mask"]
        b_fc_mask = model_state_dict["mask"][f"model.{n}.fc.bias_mask"]   # (d_model)

        # prune construct
        if not self.n_head:
            if self.d_k:
                self.n_head = w_ks_mask.shape[0] // self.d_k
            elif self.d_v:
                self.n_head = w_vs_mask.shape[0] // self.d_v
        if not self.d_k:
            self.d_k = w_ks_mask.shape[0] // self.n_head
        if not self.d_v:
            self.d_v = w_vs_mask.shape[0] // self.n_head
        assert w_qs_mask.shape == (self.n_head * self.d_k, self.d_model)
        assert b_qs_mask.shape == (self.n_head * self.d_k,)
        assert w_ks_mask.shape == (self.n_head * self.d_k, self.d_model)
        assert b_ks_mask.shape == (self.n_head * self.d_k,)
        assert w_vs_mask.shape == (self.n_head * self.d_v, self.d_model)
        assert b_vs_mask.shape == (self.n_head * self.d_v,)
        assert w_fc_mask.shape == (self.d_model, self.n_head * self.d_v)
        assert b_fc_mask.shape == (self.d_model,)

        w_qs_mask = w_qs_mask.reshape(self.n_head, self.d_k, self.d_model)
        b_qs_mask = b_qs_mask.reshape(self.n_head, self.d_k)
        w_ks_mask = w_ks_mask.reshape(self.n_head, self.d_k, self.d_model)
        b_ks_mask = b_ks_mask.reshape(self.n_head, self.d_k)
        w_vs_mask = w_vs_mask.reshape(self.n_head, self.d_v, self.d_model)
        b_vs_mask = b_vs_mask.reshape(self.n_head, self.d_v)
        w_fc_mask = w_fc_mask.reshape(self.d_model, self.n_head, self.d_v)

        n_head_index = (
            w_qs_mask.sum(dim=[1,2])
            + w_ks_mask.sum(dim=[1,2])
            + w_vs_mask.sum(dim=[1,2])
            + w_fc_mask.sum(dim=[0,2])
            + b_qs_mask.sum(dim=[1])
            + b_ks_mask.sum(dim=[1])
            + b_vs_mask.sum(dim=[1])
        ).nonzero().flatten()
        d_k_index = (
            w_qs_mask.sum(dim=[0,2])
            + w_ks_mask.sum(dim=[0,2])
            + b_qs_mask.sum(dim=[0])
            + b_ks_mask.sum(dim=[0])
        ).nonzero().flatten()
        d_v_index = (
            w_vs_mask.sum(dim=[0,2])
            + b_vs_mask.sum(dim=[0])
            + w_fc_mask.sum(dim=[0,1])
        ).nonzero().flatten()
        d_model_index = w_LN_mask.nonzero().flatten()

        self.w_qs = self.pruned_w_qs(
            n, model_state_dict,
            n_head_index, d_k_index, d_model_index)
        self.w_ks = self.pruned_w_ks(
            n, model_state_dict,
            n_head_index, d_k_index, d_model_index)
        self.w_vs = self.pruned_w_vs(
            n, model_state_dict,
            n_head_index, d_v_index, d_model_index)
        self.fc = self.pruned_fc(
            n, model_state_dict,
            n_head_index, d_v_index, d_model_index)

        self.layer_norm = self.pruned_LN(
            n, model_state_dict, d_model_index)

        self.n_head = len(n_head_index)
        self.d_k = len(d_k_index)
        self.d_v = len(d_v_index)
        self.d_model = len(d_model_index)

    def pruned_w_qs(
        self,
        n: str,
        model_state_dict: dict[str, dict[str, torch.Tensor]],
        n_head_index: torch.LongTensor,
        d_k_index: torch.LongTensor,
        d_model_index: torch.LongTensor,
    ):
        n_head = len(n_head_index)
        d_k = len(d_k_index)
        d_model = len(d_model_index)

        state_dict = OrderedDict()
        w_qs = (model_state_dict["orig"][f"model.{n}.w_qs.weight_orig"]
                .reshape(self.n_head, self.d_k, self.d_model)
                .index_select(0, n_head_index)
                .index_select(1, d_k_index)
                .index_select(2, d_model_index)
                .reshape(n_head * d_k, d_model))
        state_dict["weight"] = w_qs
        use_bias = model_state_dict["mask"][f"model.{n}.w_qs.bias_mask"].sum() > 0
        if use_bias:
            b_qs = (model_state_dict["orig"][f"model.{n}.w_qs.bias_orig"]
                    .reshape(self.n_head, self.d_k)
                    .index_select(0, n_head_index)
                    .index_select(1, d_k_index)
                    .reshape(n_head * d_k))
            state_dict["bias"] = b_qs

        new_qs = PrunedLinear(d_model, n_head * d_k, bias=use_bias, device=w_qs.device)
        new_qs.load_state_dict(state_dict)
        return new_qs

    def pruned_w_ks(
        self,
        n: str,
        model_state_dict: dict[str, dict[str, torch.Tensor]],
        n_head_index: torch.LongTensor,
        d_k_index: torch.LongTensor,
        d_model_index: torch.LongTensor,
    ):
        n_head = len(n_head_index)
        d_k = len(d_k_index)
        d_model = len(d_model_index)

        state_dict = OrderedDict()
        w_ks = (model_state_dict["orig"][f"model.{n}.w_ks.weight_orig"]
                .reshape(self.n_head, self.d_k, self.d_model)
                .index_select(0, n_head_index)
                .index_select(1, d_k_index)
                .index_select(2, d_model_index)
                .reshape(n_head * d_k, d_model))
        state_dict["weight"] = w_ks
        use_bias = model_state_dict["mask"][f"model.{n}.w_ks.bias_mask"].sum() > 0
        if use_bias:
            b_ks = (model_state_dict["orig"][f"model.{n}.w_ks.bias_orig"]
                    .reshape(self.n_head, self.d_k)
                    .index_select(0, n_head_index)
                    .index_select(1, d_k_index)
                    .reshape(n_head * d_k))
            state_dict["bias"] = b_ks

        new_ks = PrunedLinear(d_model, n_head * d_k, bias=use_bias, device=w_ks.device)
        new_ks.load_state_dict(state_dict)
        return new_ks

    def pruned_w_vs(
        self,
        n: str,
        model_state_dict: dict[str, dict[str, torch.Tensor]],
        n_head_index: torch.LongTensor,
        d_v_index: torch.LongTensor,
        d_model_index: torch.LongTensor,
    ):
        n_head = len(n_head_index)
        d_v = len(d_v_index)
        d_model = len(d_model_index)

        state_dict = OrderedDict()
        w_vs = (model_state_dict["orig"][f"model.{n}.w_vs.weight_orig"]
                .reshape(self.n_head, self.d_v, self.d_model)
                .index_select(0, n_head_index)
                .index_select(1, d_v_index)
                .index_select(2, d_model_index)
                .reshape(n_head * d_v, d_model))
        state_dict["weight"] = w_vs
        use_bias = model_state_dict["mask"][f"model.{n}.w_vs.bias_mask"].sum() > 0
        if use_bias:
            b_vs = (model_state_dict["orig"][f"model.{n}.w_vs.bias_orig"]
                    .reshape(self.n_head, self.d_v)
                    .index_select(0, n_head_index)
                    .index_select(1, d_v_index)
                    .reshape(n_head * d_v))
            state_dict["bias"] = b_vs

        new_vs = PrunedLinear(d_model, n_head * d_v, bias=use_bias, device=w_vs.device)
        new_vs.load_state_dict(state_dict)
        return new_vs

    def pruned_fc(
        self,
        n: str,
        model_state_dict: dict[str, dict[str, torch.Tensor]],
        n_head_index: torch.LongTensor,
        d_v_index: torch.LongTensor,
        d_model_index: torch.LongTensor,
    ):
        n_head = len(n_head_index)
        d_v = len(d_v_index)
        d_model = len(d_model_index)

        state_dict = OrderedDict()
        w_fc = (model_state_dict["orig"][f"model.{n}.fc.weight_orig"]
                .reshape(self.d_model, self.n_head, self.d_v)
                .index_select(0, d_model_index)
                .index_select(1, n_head_index)
                .index_select(2, d_v_index)
                .reshape(d_model, n_head * d_v))
        state_dict["weight"] = w_fc
        use_bias = model_state_dict["mask"][f"model.{n}.fc.bias_mask"].sum() > 0
        if use_bias:
            b_fc = (model_state_dict["orig"][f"model.{n}.fc.bias_orig"]
                    .index_select(0, d_model_index))
            state_dict["bias"] = b_fc

        fc = PrunedLinear(n_head * d_v, d_model, bias=use_bias, device=w_fc.device)
        fc.load_state_dict(state_dict)
        return fc

    def pruned_LN(
        self,
        n: str,
        model_state_dict: dict[str, dict[str, torch.Tensor]],
        d_model_index: torch.LongTensor,
    ):
        d_model = len(d_model_index)
        d_orig = model_state_dict["orig"][f"model.{n}.layer_norm.weight_orig"].shape[0]

        w_LN = (model_state_dict["orig"][f"model.{n}.layer_norm.weight_orig"]
                .index_select(0, d_model_index))
        b_LN = (model_state_dict["orig"][f"model.{n}.layer_norm.bias_orig"]
                .index_select(0, d_model_index))
        state_dict = OrderedDict({"weight": w_LN, "bias": b_LN})

        new_LN = PrunedLayerNorm(
            d_orig=d_orig,
            normalized_shape=d_model,
            device=b_LN.device,
        )
        new_LN.load_state_dict(state_dict)
        return new_LN


class PrunedPositionwiseFeedForward(PositionwiseFeedForward):
    """ A two-feed-forward-layer module """

    def __init__(
        self,
        d_in: int = None,
        d_hid: int = None,
        kernel_size: int = None,
        dropout: float = 0.1,
        orig: PositionwiseFeedForward = None,
    ):
        if orig:
            d_in = orig.d_in
            d_hid = orig.d_hid
            assert len(orig.w_1.kernel_size) == 1
            assert len(orig.w_2.kernel_size) == 1
            kernel_size = [orig.w_1.kernel_size[0], orig.w_2.kernel_size[0]]
            dropout = orig.dropout.p

        if d_in and d_hid and kernel_size and dropout:
            # normal constructor
            super().__init__(d_in, d_hid, kernel_size, dropout)

    def prune(
        self,
        n: str,
        model_state_dict: dict[str, dict[str, torch.Tensor]],
    ):
        w_LN_mask = model_state_dict["mask"][f"model.{n}.layer_norm.weight_mask"]   # (d_model)
        b_LN_mask = model_state_dict["mask"][f"model.{n}.layer_norm.bias_mask"]   # (d_model)
        self.d_in = self.d_in or w_LN_mask.shape[0]
        assert w_LN_mask.shape == (self.d_in,)
        assert b_LN_mask.shape == (self.d_in,)
        d_in_index = w_LN_mask.nonzero().flatten()

        # conv1d: d_out * d_in * d_kernel
        w_1_mask = model_state_dict["mask"][f"model.{n}.w_1.weight_mask"]
        b_1_mask = model_state_dict["mask"][f"model.{n}.w_1.bias_mask"]
        w_2_mask = model_state_dict["mask"][f"model.{n}.w_2.weight_mask"]
        b_2_mask = model_state_dict["mask"][f"model.{n}.w_2.bias_mask"]
        assert w_1_mask.shape[:2] == (self.d_hid, self.d_in)
        assert b_1_mask.shape == (self.d_hid,)
        assert w_2_mask.shape[:2] == (self.d_in, self.d_hid)
        assert b_2_mask.shape == (self.d_in,)

        d_hid_index = (
            w_1_mask.sum(dim=[1,2])
            + b_1_mask
            + w_2_mask.sum(dim=[0,2])
        ).nonzero().flatten()

        self.w_1 = self.pruned_w_1(n, model_state_dict, d_in_index,
                                    d_hid_index)
        self.w_2 = self.pruned_w_2(n, model_state_dict, d_in_index,
                                    d_hid_index)
        self.layer_norm = self.pruned_LN(n, model_state_dict, d_in_index)

        self.d_in = len(d_in_index)
        self.d_hid = len(d_hid_index)

    def pruned_w_1(
        self,
        n: str,
        model_state_dict: dict[str, dict[str, torch.Tensor]],
        d_in_index: torch.LongTensor,
        d_hid_index: torch.LongTensor,
    ):
        d_in = len(d_in_index)
        d_hid = len(d_hid_index)

        state_dict = OrderedDict()
        w_1 = (model_state_dict["orig"][f"model.{n}.w_1.weight_orig"]
               .index_select(0, d_hid_index)
               .index_select(1, d_in_index))
        state_dict["weight"] = w_1
        use_bias = model_state_dict["mask"][f"model.{n}.w_1.bias_mask"].sum() > 0
        if use_bias:
            b_1 = (model_state_dict["orig"][f"model.{n}.w_1.bias_orig"]
                    .index_select(0, d_hid_index))
            state_dict["bias"] = b_1

        new_1 = PrunedConv1d(
            d_in, d_hid,
            bias=use_bias,
            kernel_size=self.w_1.kernel_size[0],
            padding=self.w_1.padding[0],
            device=w_1.device
        )
        new_1.load_state_dict(state_dict)
        return new_1

    def pruned_w_2(
        self,
        n: str,
        model_state_dict: dict[str, dict[str, torch.Tensor]],
        d_in_index: torch.LongTensor,
        d_hid_index: torch.LongTensor,
    ):
        d_in = len(d_in_index)
        d_hid = len(d_hid_index)

        state_dict = OrderedDict()
        w_2 = (model_state_dict["orig"][f"model.{n}.w_2.weight_orig"]
               .index_select(0, d_in_index)
               .index_select(1, d_hid_index))
        state_dict["weight"] = w_2
        use_bias = model_state_dict["mask"][f"model.{n}.w_2.bias_mask"].sum() > 0
        if use_bias:
            b_2 = (model_state_dict["orig"][f"model.{n}.w_2.bias_orig"]
                    .index_select(0, d_in_index))
            state_dict["bias"] = b_2

        new_2 = PrunedConv1d(
            d_hid, d_in,
            bias=use_bias,
            kernel_size=self.w_2.kernel_size[0],
            padding=self.w_2.padding[0],
            device=w_2.device
        )
        new_2.load_state_dict(state_dict)
        return new_2

    def pruned_LN(
        self,
        n: str,
        model_state_dict: dict[str, dict[str, torch.Tensor]],
        d_in_index: torch.LongTensor,
    ):
        d_in = len(d_in_index)
        d_orig = model_state_dict["orig"][f"model.{n}.layer_norm.weight_orig"].shape[0]

        w_LN = (model_state_dict["orig"][f"model.{n}.layer_norm.weight_orig"]
                .index_select(0, d_in_index))
        b_LN = (model_state_dict["orig"][f"model.{n}.layer_norm.bias_orig"]
                .index_select(0, d_in_index))
        state_dict = OrderedDict({"weight": w_LN, "bias": b_LN})

        new_LN = PrunedLayerNorm(
            d_orig=d_orig,
            normalized_shape=d_in,
            device=b_LN.device,
        )
        new_LN.load_state_dict(state_dict)
        return new_LN
