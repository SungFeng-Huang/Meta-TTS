import os
import json
import torch
import torch.nn as nn
import numpy as np

from collections import OrderedDict
from pytorch_lightning.utilities.cli import instantiate_class

from utils.tools import get_mask_from_lengths, pad
from .utils import FeatureLevel, QuantizationType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self,
                 predictor: dict,
                 pitch_feature_level: FeatureLevel,
                 energy_feature_level: FeatureLevel,
                 pitch_quantization: QuantizationType,
                 energy_quantization: QuantizationType,
                 n_bins: int,
                 d_hidden: int,
                 stats_file: str,
                 pitch_normalization: bool,
                 energy_normalization: bool,
                 pitch_log: bool,
                 energy_log: bool,
                 **kwargs):
        """
        Args:
            predictor: VariancePredictor config.
                Including "args" and "init" for `instantiate_class`.
            pitch_feature_level: preprocess_config["preprocessing"]["pitch"]["feature"]
            energy_feature_level: preprocess_config["preprocessing"]["energy"]["feature"]
            pitch_quantization: model_config["variance_embedding"]["pitch_quantization"]
            energy_quantization: model_config["variance_embedding"]["energy_quantization"]
            n_bins: model_config["variance_embedding"]["n_bins"]
            d_hidden: model_config["transformer"]["encoder_hidden"]
            stats_file: f'{preprocess_config["path"]["preprocessed_path"]}/stats.json'
            pitch_normalization: preprocess_config["preprocessing"]["pitch"]["normalization"]
            energy_normalization: preprocess_config["preprocessing"]["energy"]["normalization"]
            pitch_log: preprocess_config["preprocessing"]["pitch"]["log"]
            energy_log: preprocess_config["preprocessing"]["energy"]["log"]
        """
        super().__init__()
        if not FeatureLevel.supported_level(pitch_feature_level):
            raise ValueError
        if not FeatureLevel.supported_level(energy_feature_level):
            raise ValueError

        if not QuantizationType.supported_type(pitch_quantization):
            raise ValueError
        if not QuantizationType.supported_type(energy_quantization):
            raise ValueError

        self.duration_predictor = instantiate_class(**predictor)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = instantiate_class(**predictor)
        self.energy_predictor = instantiate_class(**predictor)
        self.pitch_feature_level = pitch_feature_level
        self.energy_feature_level = energy_feature_level

        with open(stats_file) as f:
            stats = json.load(f)
            if pitch_normalization:
                pitch_min = (stats["pitch"]["min"] - stats["pitch"]["mean"]) / stats["pitch"]["std"]
                pitch_max = (stats["pitch"]["max"] - stats["pitch"]["mean"]) / stats["pitch"]["std"]
            elif pitch_log:
                pitch_min = np.log(stats["pitch"]["min"] + 1e-12)
                pitch_max = np.log(stats["pitch"]["max"] + 1e-12)
            else:
                pitch_min, pitch_max = stats["pitch"]["min"], stats["pitch"]["max"]
            if energy_normalization:
                energy_min = (stats["energy"]["min"] - stats["energy"]["mean"]) / stats["energy"]["std"]
                energy_max = (stats["energy"]["max"] - stats["energy"]["mean"]) / stats["energy"]["std"]
            elif energy_log:
                energy_min = np.log(stats["energy"]["min"] + 1e-12)
                energy_max = np.log(stats["energy"]["max"] + 1e-12)
            else:
                energy_min, energy_max = stats["energy"]["min"], stats["energy"]["max"]

        if pitch_quantization == QuantizationType.LOG:
            if pitch_normalization or pitch_log:
                raise ValueError
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min+1e-12), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        elif pitch_quantization == QuantizationType.LINEAR:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == QuantizationType.LOG:
            if energy_normalization or energy_log:
                raise ValueError
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min+1e-12), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        elif energy_quantization == QuantizationType.LINEAR:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(n_bins, d_hidden)
        self.energy_embedding = nn.Embedding(n_bins, d_hidden)

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):

        log_duration_prediction = self.duration_predictor(x, src_mask)
        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, src_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, e_control
            )
            x = x + energy_embedding

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, e_control
            )
            x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor """

    def __init__(self,
                 input_size: int,
                 filter_size: int,
                 kernel: int,
                 conv_output_size: int,
                 dropout: float,
                 **kwargs):
        """
        Args:
            input_size: model_config["transformer"]["encoder_hidden"]
            filter_size: model_config["variance_predictor"]["filter_size"]
            kernel: model_config["variance_predictor"]["kernel_size"]
            conv_output_size: model_config["variance_predictor"]["filter_size"]
            dropout: model_config["variance_predictor"]["dropout"]
        """
        super().__init__()

        self.input_size = input_size
        self.filter_size = filter_size
        self.kernel = kernel
        self.conv_output_size = conv_output_size
        self.dropout = dropout

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x
