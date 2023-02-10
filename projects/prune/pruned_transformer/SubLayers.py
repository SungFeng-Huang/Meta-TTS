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
            return torch.zeros(
                *input.shape[:-1], self.out_features,
                device=input.device,
            ) + self.bias


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
        # n: str = None,
        # model_state_dict: dict[str, dict[str, torch.Tensor]] = None,
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
        # elif n and model_state_dict:
        #     # prune constructor
        #     assert n_head or d_k or d_v
        #     torch.nn.Module.__init__(self)
        #     if n_head:
        #         self.n_head = n_head
        #     if d_k:
        #         self.d_k = d_k
        #     if d_v:
        #         self.d_v = d_v
        #     self.prune(n, model_state_dict)
        #     self.attention = ScaledDotProductAttention(temperature=16)
        #     self.dropout = nn.Dropout(dropout)

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

        # if len(n_head_index) == 0:
        #     self.n_head = self.d_k = self.d_v = 0
        #     self.keep_k = self.keep_v = False
        # else:
        #     self.keep_k = len(d_k_index) > 0
        #     self.keep_v = len(d_v_index) > 0
        #
        # if self.keep_k and self.keep_v:
        #     # only partial prune
        self.w_qs = self.pruned_w_qs(
            n, model_state_dict,
            n_head_index, d_k_index, d_model_index)
        self.w_ks = self.pruned_w_qs(
            n, model_state_dict,
            n_head_index, d_k_index, d_model_index)
        self.w_vs = self.pruned_w_qs(
            n, model_state_dict,
            n_head_index, d_v_index, d_model_index)
        self.fc = self.pruned_fc(
            n, model_state_dict,
            n_head_index, d_v_index, d_model_index)

        # elif self.keep_v:
        #     # d_k == 0
        #     # remove w_qs, w_ks -> no attention (simply average)
        #     del self.w_qs, self.w_ks
        #     del self.attention
        #     self.attention = self.pruned_attention
        #     self.w_vs = self.pruned_w_qs(n, model_state_dict, n_head_index,
        #                                  d_v_index, d_model_index)
        #     self.fc = self.pruned_fc(n, model_state_dict, n_head_index,
        #                              d_v_index, d_model_index)
        #
        # else:
        #     # d_k == d_v == 0 -> n_head == 0
        #     # self.keep_v == False -> set self.keep_k = False
        #     self.keep_k = False
        #
        #     # remove w_vs, fc -> also remove w_qs, w_ks
        #     del self.w_qs, self.w_ks, self.w_vs, self.fc
        #
        #     if b_fc_mask.sum() > 0:
        #         self.fc_bias = torch.nn.Parameter(
        #             model_state_dict["orig"][f"model.{n}.fc.bias_orig"]
        #             .index_select(0, d_model_index)
        #         )

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
        # if b_qs_mask.sum() > 0:
        b_qs = (model_state_dict["orig"][f"model.{n}.w_qs.bias_orig"]
                .reshape(self.n_head, self.d_k)
                .index_select(0, n_head_index)
                .index_select(1, d_k_index)
                .reshape(n_head * d_k))
        if model_state_dict["mask"][f"model.{n}.w_qs.bias_mask"].sum() == 0:
            b_qs *= 0
        state_dict["bias"] = b_qs

        new_qs = PrunedLinear(d_model, n_head * d_k,
                              device=b_qs.device)
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
        # if b_ks_mask.sum() > 0:
        b_ks = (model_state_dict["orig"][f"model.{n}.w_ks.bias_orig"]
                .reshape(self.n_head, self.d_k)
                .index_select(0, n_head_index)
                .index_select(1, d_k_index)
                .reshape(n_head * d_k))
        if model_state_dict["mask"][f"model.{n}.w_ks.bias_mask"].sum() == 0:
            b_ks *= 0
        state_dict["bias"] = b_ks

        new_ks = PrunedLinear(d_model, n_head * d_k,
                              device=b_ks.device)
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
        # if b_vs_mask.sum() > 0:
        b_vs = (model_state_dict["orig"][f"model.{n}.w_vs.bias_orig"]
                .reshape(self.n_head, self.d_v)
                .index_select(0, n_head_index)
                .index_select(1, d_v_index)
                .reshape(n_head * d_v))
        if model_state_dict["mask"][f"model.{n}.w_vs.bias_mask"].sum() == 0:
            b_vs *= 0
        state_dict["bias"] = b_vs

        new_vs = PrunedLinear(d_model, n_head * d_v,
                              device=b_vs.device)
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
        # if b_fc_mask.sum() > 0:
        b_fc = (model_state_dict["orig"][f"model.{n}.fc.bias_orig"]
                .index_select(0, d_model_index))
        if model_state_dict["mask"][f"model.{n}.fc.bias_mask"].sum() == 0:
            b_fc *= 0
        state_dict["bias"] = b_fc

        fc = PrunedLinear(n_head * d_v, d_model,
                          device=b_fc.device)
        fc.load_state_dict(state_dict)
        return fc

    def pruned_LN(
        self,
        n: str,
        model_state_dict: dict[str, dict[str, torch.Tensor]],
        d_model_index: torch.LongTensor,
    ):
        d_model = len(d_model_index)

        w_LN = (model_state_dict["orig"][f"model.{n}.layer_norm.weight_orig"]
                .index_select(0, d_model_index))
        b_LN = (model_state_dict["orig"][f"model.{n}.layer_norm.bias_orig"]
                .index_select(0, d_model_index))
        state_dict = OrderedDict({"weight": w_LN, "bias": b_LN})

        new_LN = nn.LayerNorm(d_model,
                              device=b_LN.device)
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
        # n: str = None,
        # model_state_dict: dict[str, dict[str, torch.Tensor]] = None,
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
        # elif n and model_state_dict:
        #     # prune constructor
        #     torch.nn.Module.__init__(self)
            # self.prune(n, model_state_dict)
            # self.dropout = nn.Dropout(dropout)

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

        # self.keep_w = len(d_hid_index) > 0
        # if self.keep_w:
        self.w_1 = self.pruned_w_1(n, model_state_dict, d_in_index,
                                    d_hid_index)
        self.w_2 = self.pruned_w_2(n, model_state_dict, d_in_index,
                                    d_hid_index)
        # else:
        #     b_2_mask = model_state_dict["mask"][f"model.{n}.w_2.bias_mask"]   # (d_model)
        #     if b_2_mask.sum() > 0:
        #         self.b_2 = torch.nn.Parameter(
        #             model_state_dict["orig"][f"model.{n}.w_2.bias_orig"]
        #             .index_select(0, d_in_index)
        #         )
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
        # if b_1_mask.sum() > 0:
        b_1 = (model_state_dict["orig"][f"model.{n}.w_1.bias_orig"]
                .index_select(0, d_hid_index))
        if model_state_dict["mask"][f"model.{n}.w_1.bias_mask"].sum() == 0:
            b_1 *= 0
        state_dict["bias"] = b_1

        new_1 = nn.Conv1d(d_in, d_hid,
                          kernel_size=self.w_1.kernel_size[0],
                          padding=self.w_1.padding[0],
                          # bias=(b_1_mask.sum() > 0),
                          device=b_1.device)
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
        # if b_2_mask.sum() > 0:
        b_2 = (model_state_dict["orig"][f"model.{n}.w_2.bias_orig"]
                .index_select(0, d_in_index))
        if model_state_dict["mask"][f"model.{n}.w_2.bias_mask"].sum() == 0:
            b_2 *= 0
        state_dict["bias"] = b_2

        new_2 = nn.Conv1d(d_hid, d_in,
                          kernel_size=self.w_2.kernel_size[0],
                          padding=self.w_2.padding[0],
                          # bias=(b_2_mask.sum() > 0),
                          device=b_2.device)
        new_2.load_state_dict(state_dict)
        return new_2

    def pruned_LN(
        self,
        n: str,
        model_state_dict: dict[str, dict[str, torch.Tensor]],
        d_in_index: torch.LongTensor,
    ):
        d_in = len(d_in_index)

        w_LN = (model_state_dict["orig"][f"model.{n}.layer_norm.weight_orig"]
                .index_select(0, d_in_index))
        b_LN = (model_state_dict["orig"][f"model.{n}.layer_norm.bias_orig"]
                .index_select(0, d_in_index))
        state_dict = OrderedDict({"weight": w_LN, "bias": b_LN})

        new_LN = nn.LayerNorm(d_in,
                              device=b_LN.device)
        new_LN.load_state_dict(state_dict)
        return new_LN

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        if self.w_1.out_channels > 0:
            output = self.w_2(F.relu(self.w_1(output)))
        else:
            output = torch.zeros(
                output.shape[0], self.w_2.out_channels, output.shape[2],
                device=output.device
            ) + self.w_2.bias[None, :, None]
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output

    # def forward(self, x):
    #     residual = x
    #     output = x.transpose(1, 2)
    #     if self.keep_w:
    #         output = self.w_2(F.relu(self.w_1(output)))
    #     else:
    #         output = torch.zeros_like(output) + self.b_2
    #     output = output.transpose(1, 2)
    #     output = self.dropout(output)
    #     output = self.layer_norm(output + residual)
    #
    #     return output

