from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

from src.transformer.Layers import ConvNorm, PostNet
from lightning.model.modules import VariancePredictor, Conv
from .SubLayers import PrunedLinear

class PrunedConvNorm(ConvNorm):

    def forward(self, signal):
        if self.conv.in_channels and self.conv.out_channels:
            # d_in > 0, d_out > 0
            conv_signal = self.conv(signal)
        elif self.conv.out_channels:
            # d_in == 0, d_out > 0
            # conv would ourput wrong size
            conv_signal = torch.zeros(
                signal.shape[0], self.conv.out_channels, signal.shape[2],
                device=signal.device,
            ) + self.conv.bias[None, :, None]
        else:
            # d_out == 0
            conv_signal = torch.randn(
                signal.shape[0], 0, signal.shape[2],
                device=signal.device,
            )

        return conv_signal

class PrunedPostNet(PostNet):
    """
    PostNet: Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(
        self,
        n_mel_channels: int = 80,
        postnet_embedding_dim: int = 512,
        postnet_kernel_size: int = 5,
        postnet_n_convolutions: int = 5,
        orig: PostNet = None,
        # n: str = None,
        # model_state_dict: dict[str, dict[str, torch.Tensor]] = None,
    ):
        assert postnet_kernel_size or orig
        if orig:
            n_mel_channels = orig.n_mel_channels
            postnet_embedding_dim = orig.d_embedding
            postnet_kernel_size = orig.convolutions[0][0].conv.kernel_size[0]
            postnet_n_convolutions = len(orig.convolutions)

        if n_mel_channels and postnet_embedding_dim and postnet_kernel_size and postnet_n_convolutions:
            # normal constructor
            super().__init__(n_mel_channels, postnet_embedding_dim,
                            postnet_kernel_size, postnet_n_convolutions)
        # elif n and model_state_dict:
        #     # prune constructor
        #     torch.nn.Module.__init__(self)
        #     self.convolutions = nn.ModuleList()
        #     self.prune(n, model_state_dict)
        #

    def prune(
        self,
        n: str,
        model_state_dict: dict[str, dict[str, torch.Tensor]],
    ):
        # NOTE: self.n_mel_channels would not be pruned
        d_conv_indices = []
        for i in range(len(self.convolutions)):
            # Deal with the hidden units between layer i and layer i+1
            conv_0_w_mask = model_state_dict["mask"][f"model.{n}.convolutions.{i}.0.conv.weight_mask"]
            BN_0_mask = model_state_dict["mask"][f"model.{n}.convolutions.{i}.1.weight_mask"]

            if len(d_conv_indices) == 0:
                d_in_index = conv_0_w_mask.sum(dim=[0,2]).nonzero().flatten()
                d_conv_indices.append(d_in_index)
            else:
                d_in_index = d_conv_indices[-1]

            d_out_index = BN_0_mask.nonzero().flatten()

            self.convolutions[i][0] = self.pruned_conv(
                n, model_state_dict, i, d_in_index, d_out_index)
            self.convolutions[i][1] = self.pruned_BN(
                n, model_state_dict, i, d_out_index)
            d_conv_indices.append(d_out_index)

    def pruned_conv(
        self,
        n: str,
        model_state_dict: dict[str, dict[str, torch.Tensor]],
        i: int,
        d_in_index: torch.LongTensor,
        d_out_index: torch.LongTensor,
    ):
        d_in = len(d_in_index)
        d_out = len(d_out_index)

        state_dict = OrderedDict()
        conv_w = (model_state_dict["orig"][f"model.{n}.convolutions.{i}.0.conv.weight_orig"]
                  .index_select(0, d_out_index)
                  .index_select(1, d_in_index))
        state_dict["weight"] = conv_w
        conv_b = (model_state_dict["orig"][f"model.{n}.convolutions.{i}.0.conv.bias_orig"]
                    .index_select(0, d_out_index))
        if model_state_dict["mask"][f"model.{n}.convolutions.{i}.0.conv.bias_mask"].sum() == 0:
            conv_b *= 0
        state_dict["bias"] = conv_b

        conv = PrunedConvNorm(
            d_in, d_out,
            kernel_size=self.convolutions[i][0].conv.kernel_size[0],
            stride=1,
            padding=self.convolutions[i][0].conv.padding[0],
            dilation=1,
        ).to(device=conv_b.device)
        conv.conv.load_state_dict(state_dict)
        return conv

    def pruned_BN(
        self,
        n: str,
        model_state_dict: dict[str, dict[str, torch.Tensor]],
        i: int,
        d_out_index: torch.LongTensor,
    ):
        d_out = len(d_out_index)

        w_BN = (model_state_dict["orig"][f"model.{n}.convolutions.{i}.1.weight_orig"]
                .index_select(0, d_out_index))
        b_BN = (model_state_dict["orig"][f"model.{n}.convolutions.{i}.1.bias_orig"]
                .index_select(0, d_out_index))
        rm_BN = (model_state_dict["others"][f"model.{n}.convolutions.{i}.1.running_mean"]
                 .index_select(0, d_out_index))
        rv_BN = (model_state_dict["others"][f"model.{n}.convolutions.{i}.1.running_var"]
                 .index_select(0, d_out_index))
        num_batches_tracked = model_state_dict["others"][f"model.{n}.convolutions.{i}.1.num_batches_tracked"]
        state_dict = OrderedDict({
            "weight": w_BN,
            "bias": b_BN,
            "running_mean": rm_BN,
            "running_var": rv_BN,
            "num_batches_tracked": num_batches_tracked,
        })

        new_BN = nn.BatchNorm1d(d_out,
                                device=w_BN.device)
        new_BN.load_state_dict(state_dict)
        return new_BN

    # def forward(self, x: torch.Tensor):
    #     x = x.contiguous().transpose(1, 2)

    #     for i in range(len(self.convolutions) - 1):
    #         if self.convolutions[i][0].in_channels and self.convolutions[i][0].out_channels:
    #             # d_in > 0, d_out > 0
    #             x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
    #         elif self.convolutions[i][0].out_channels:
    #             # d_in = 0, d_out > 0
    #             # conv would output wrong size
    #             x = torch.zeros(
    #                 x.shape[0], self.convolutions[i][0].out_channels, x.shape[2],
    #                 device=x.device,
    #             ) + self.convolutions[i][0].bias[None, :, None]
    #             x = F.dropout(torch.tanh(self.convolutions[i][1](x)), 0.5, self.training)
    #         else:
    #             # d_in ? 0, d_out = 0
    #             # conv would raise RuntimeError, BN would raise IndexError
    #             # directly output (B, 0, L)-sized tensor
    #             x = torch.randn(x.shape[0], 0, x.shape[2], device=x.device)

    #     # assert output dim > 0
    #     if self.convolutions[-1][0].in_channels:
    #         # d_in > 0, d_out > 0
    #         x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
    #     else:
    #         # d_in = 0, d_out > 0
    #         # conv would output wrong size
    #         x = torch.zeros(
    #             x.shape[0], self.convolutions[-1][0].out_channels, x.shape[2],
    #             device=x.device,
    #         ) + self.convolutions[-1][0].bias[None, :, None]
    #         x = F.dropout(torch.tanh(self.convolutions[-1][1](x)), 0.5, self.training)

    #     x = x.contiguous().transpose(1, 2)
    #     return x


class PrunedVariancePredictor(VariancePredictor):
    """ Duration, Pitch and Energy Predictor """

    def __init__(
        self,
        model_config: dict[str, dict] = None,
        orig: VariancePredictor = None
    ):
        assert model_config is not None or orig is not None
        if orig:
            model_config = {
                "transformer": {
                    "encoder_hidden": orig.input_size,
                },
                "variance_predictor": {
                    "filter_size": orig.filter_size,
                    "kernel_size": orig.kernel,
                    "dropout": orig.dropout,
                }
            }
        super().__init__(model_config)

    def prune(
        self,
        n: str,
        model_state_dict: dict[str, dict[str, torch.Tensor]],
    ):
        conv_1_w_mask = model_state_dict["mask"][f"model.{n}.conv_layer.conv1d_1.conv.weight_mask"]
        conv_1_b_mask = model_state_dict["mask"][f"model.{n}.conv_layer.conv1d_1.conv.bias_mask"]
        LN_1_mask = model_state_dict["mask"][f"model.{n}.conv_layer.layer_norm_1.weight_mask"]
        conv_2_w_mask = model_state_dict["mask"][f"model.{n}.conv_layer.conv1d_2.conv.weight_mask"]
        conv_2_b_mask = model_state_dict["mask"][f"model.{n}.conv_layer.conv1d_2.conv.bias_mask"]
        LN_2_mask = model_state_dict["mask"][f"model.{n}.conv_layer.layer_norm_2.weight_mask"]
        lin_w_mask = model_state_dict["mask"][f"model.{n}.linear_layer.weight_mask"]
        lin_b_mask = model_state_dict["mask"][f"model.{n}.linear_layer.bias_mask"]

        d_conv_0_index = conv_1_w_mask.sum(dim=[0,2]).nonzero().flatten()
        d_conv_1_index = LN_1_mask.nonzero().flatten()
        d_conv_2_index = LN_2_mask.nonzero().flatten()
        d_lin_index = lin_w_mask.sum(dim=[1]).nonzero().flatten()

        self.conv_layer.conv1d_1 = self.pruned_conv(n, model_state_dict, 1, d_conv_0_index, d_conv_1_index)
        self.conv_layer.layer_norm_1 = self.pruned_LN(n, model_state_dict, 1, d_conv_1_index)
        self.conv_layer.conv1d_2 = self.pruned_conv(n, model_state_dict, 2, d_conv_1_index, d_conv_2_index)
        self.conv_layer.layer_norm_2 = self.pruned_LN(n, model_state_dict, 2, d_conv_2_index)
        self.linear_layer = self.pruned_linear(n, model_state_dict, d_conv_2_index, d_lin_index)
    
    def pruned_conv(
        self,
        n: str,
        model_state_dict: dict[str, dict[str, torch.Tensor]],
        i: int,
        d_in_index: torch.LongTensor,
        d_out_index: torch.LongTensor,
    ):
        d_in = len(d_in_index)
        d_out = len(d_out_index)

        state_dict = OrderedDict()
        conv_w = (model_state_dict["orig"][f"model.{n}.conv_layer.conv1d_{i}.conv.weight_orig"]
                  .index_select(0, d_out_index)
                  .index_select(1, d_in_index))
        state_dict["weight"] = conv_w
        conv_b = (model_state_dict["orig"][f"model.{n}.conv_layer.conv1d_{i}.conv.bias_orig"]
                    .index_select(0, d_out_index))
        if model_state_dict["mask"][f"model.{n}.conv_layer.conv1d_{i}.conv.bias_mask"].sum() == 0:
            conv_b *= 0
        state_dict["bias"] = conv_b

        conv = PrunedConv(
            d_in, d_out,
            kernel_size=getattr(self.conv_layer, f"conv1d_{i}").conv.kernel_size[0],
            stride=1,
            padding=getattr(self.conv_layer, f"conv1d_{i}").conv.padding[0],
            dilation=1,
        ).to(device=conv_b.device)
        conv.conv.load_state_dict(state_dict)
        return conv

    def pruned_LN(
        self,
        n: str,
        model_state_dict: dict[str, dict[str, torch.Tensor]],
        i: int,
        d_in_index: torch.LongTensor,
    ):
        d_in = len(d_in_index)

        w_LN = (model_state_dict["orig"][f"model.{n}.conv_layer.layer_norm_{i}.weight_orig"]
                .index_select(0, d_in_index))
        b_LN = (model_state_dict["orig"][f"model.{n}.conv_layer.layer_norm_{i}.bias_orig"]
                .index_select(0, d_in_index))
        state_dict = OrderedDict({"weight": w_LN, "bias": b_LN})

        new_LN = nn.LayerNorm(d_in,
                              device=b_LN.device)
        new_LN.load_state_dict(state_dict)
        return new_LN

    def pruned_linear(
        self,
        n: str,
        model_state_dict: dict[str, dict[str, torch.Tensor]],
        d_in_index: torch.LongTensor,
        d_out_index: torch.LongTensor,
    ):
        d_in = len(d_in_index)
        d_out = len(d_out_index)
        assert d_out == 1

        state_dict = OrderedDict()
        w = (model_state_dict["orig"][f"model.{n}.linear_layer.weight_orig"]
             .index_select(0, d_out_index)
             .index_select(1, d_in_index))
        state_dict["weight"] = w
        b = (model_state_dict["orig"][f"model.{n}.linear_layer.bias_orig"]
             .index_select(0, d_out_index))
        if model_state_dict["mask"][f"model.{n}.linear_layer.bias_mask"].sum() == 0:
            b *= 0
        state_dict["bias"] = b

        linear = PrunedLinear(d_in, d_out,
                              device=b.device)
        linear.load_state_dict(state_dict)
        return linear


class PrunedConv(Conv):

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        if self.conv.in_channels and self.conv.out_channels:
            # d_in > 0, d_out > 0
            x = self.conv(x)
        elif self.conv.out_channels:
            # d_in == 0, d_out > 0
            # conv would ourput wrong size
            x = torch.zeros(
                x.shape[0], self.conv.out_channels, x.shape[2],
                device=x.device,
            ) + self.conv.bias[None, :, None]
        else:
            # d_out == 0
            x = torch.randn(x.shape[0], 0, x.shape[2], device=x.device)
        x = x.contiguous().transpose(1, 2)

        return x
