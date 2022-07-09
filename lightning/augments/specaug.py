# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
# Copied from S3PRL: https://github.com/s3prl/s3prl/blob/master/s3prl/downstream/specaug.py
#   FileName     [ dataset.py ]
#   Synopsis     [ the phone dataset ]
#   Author       [ S3PRL, Xuankai Chang ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""
# Adaptive_SpecAugment Author: ShampooWang, cornliu

###############
# IMPORTATION #
###############
import os
import random
#-------------#
import pandas as pd
#-------------#
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
#-------------#
# import torchaudio


DEFAULT_TIME_WARP_MODE = "bicubic"

class SpecAug(torch.nn.Module):
    """SpecAug"""
    def __init__(
        self,
        apply_time_warp=True,
        time_warp_window=5,
        time_warp_mode="bicubic",
        apply_freq_mask=True,
        freq_mask_width_range=(0,20),
        num_freq_mask=2,
        apply_time_mask=True,
        time_mask_width_range=(0,100),
        num_time_mask=2,
        adaptive_number_ratio = 0.04,
        adaptive_size_ratio = 0.04,
        max_n_time_masks = 20,
        adaptive=False
    ):
        assert any([apply_time_warp, apply_freq_mask, apply_time_mask])

        super(SpecAug, self).__init__()
        self.apply_time_warp = apply_time_warp
        self.apply_freq_mask = apply_freq_mask
        self.apply_time_mask = apply_time_mask

        if apply_time_warp:
            self.time_warp = TimeWarp(window=time_warp_window, mode=time_warp_mode)
        else:
            self.time_warp = None

        if apply_freq_mask:
            self.freq_mask = MaskAlongAxis(
                dim="freq",
                mask_width_range=freq_mask_width_range,
                num_mask=num_freq_mask,
            )
        else:
            self.freq_mask = None

        if apply_time_mask:
            self.time_mask = MaskAlongAxis(
                dim="time",
                mask_width_range=time_mask_width_range,
                num_mask=num_time_mask,
                adaptive=adaptive,
                adaptive_number_ratio=adaptive_number_ratio,
                adaptive_size_ratio=adaptive_size_ratio,
                max_n_time_masks=max_n_time_masks
            )
        else:
            self.time_mask = None

    def apply_specaug(self, x, x_lengths=None):
        if self.time_warp is not None:
            x, x_lengths = self.time_warp(x, x_lengths)
        if self.freq_mask is not None:
            x, x_lengths = self.freq_mask(x, x_lengths)
        if self.time_mask is not None:
            x, x_lengths = self.time_mask(x, x_lengths)
        return x, x_lengths

    def forward(self, xs, x_lengths=None):
        """Forward SpecAug.

        Args:
            xs: list of features [(T, D)] x batchsize
            x_lengths: length of features
        Return:
            list of features[(T, D)] x batchsize
        """
        assert len(xs[0].size()) == 2
        if x_lengths is None:
            x_lengths = torch.LongTensor([x.size(0) for x in xs])

        batchsize, max_len, dim = len(x_lengths), torch.max(x_lengths).item(), xs[0].size(1)
        xs_pad = xs[0].new_zeros((batchsize, max_len, dim))
        for i, x in enumerate(xs):
            xs_pad[i, :x_lengths[i]] = x

        xs_pad, _ = self.apply_specaug(xs_pad, x_lengths)

        xs = [xs_pad[i, :xs[i].size(0)] for i in range(batchsize)]
        return xs, x_lengths


class TimeWarp(torch.nn.Module):
    """Time warping using torch.interpolate.
    Args:
        window: time warp parameter
        mode: Interpolate mode
    """

    def __init__(self, window=80, mode=DEFAULT_TIME_WARP_MODE):
        super().__init__()
        self.window = window
        self.mode = mode

    def extra_repr(self):
        return f"window={self.window}, mode={self.mode}"

    def time_warp(self, x):
        org_size = x.size()
        if x.dim() == 3:
            # x: (Batch, Time, Freq) -> (Batch, 1, Time, Freq)
            x = x[:, None]

        t = x.shape[2]
        if t - self.window <= self.window:
            return x.view(*org_size)

        center = torch.randint(self.window, t - self.window, (1,))[0]
        warped = torch.randint(center - self.window, center + self.window, (1,))[0] + 1

        # left: (Batch, Channel, warped, Freq)
        # right: (Batch, Channel, time - warped, Freq)
        left = torch.nn.functional.interpolate(
            x[:, :, :center], (warped, x.shape[3]), mode=self.mode, align_corners=False
        )
        right = torch.nn.functional.interpolate(
            x[:, :, center:], (t - warped, x.shape[3]), mode=self.mode, align_corners=False
        )

        if x.requires_grad:
            x = torch.cat([left, right], dim=-2)
        else:
            x[:, :, :warped] = left
            x[:, :, warped:] = right

        return x.view(*org_size)

    def forward(self, x, x_lengths=None):
        """Forward function.
        Args:
            x: (Batch, Time, Freq)
            x_lengths: (Batch,)
        """
        ys = x.new_zeros(x.size())
        for i in range(x.size(0)):
            _y = self.time_warp(
                x[i][None, : x_lengths[i]],
            )[0]
            ys[i, : x_lengths[i]] = _y

        return ys, x_lengths


class MaskAlongAxis(torch.nn.Module):
    def __init__(
        self,
        mask_width_range=(0, 30),
        num_mask=2,
        dim="time",
        replace_with_zero=True,
        adaptive_number_ratio = 0.04,
        adaptive_size_ratio = 0.04,
        max_n_time_masks = 20,
        adaptive = False
    ):
        if isinstance(mask_width_range, int):
            mask_width_range = (0, mask_width_range)
        if len(mask_width_range) != 2:
            raise TypeError(
                f"mask_width_range must be a tuple of int and int values: "
                f"{mask_width_range}",
            )

        assert mask_width_range[1] > mask_width_range[0]
        if isinstance(dim, str):
            if dim == "time":
                dim = 1
            elif dim == "freq":
                dim = 2
            else:
                raise ValueError("dim must be int, 'time' or 'freq'")
        if dim == 1:
            self.mask_axis = "time"
        elif dim == 2:
            self.mask_axis = "freq"
        else:
            self.mask_axis = "unknown"

        super().__init__()
        self.mask_width_range = mask_width_range
        self.num_mask = num_mask
        self.dim = dim
        self.replace_with_zero = replace_with_zero

        ###############################################
        self.adaptive = adaptive
        self.adaptive_number_ratio = adaptive_number_ratio
        self.adaptive_size_ratio = adaptive_size_ratio
        self.max_n_time_masks = max_n_time_masks
        ###############################################

    def mask_along_axis(self, spec, spec_lengths):
        org_size = spec.size()
        if spec.dim() == 4:
            # spec: (Batch, Channel, Length, Freq) -> (Batch * Channel, Length, Freq)
            spec = spec.view(-1, spec.size(2), spec.size(3))

        B = spec.shape[0]

        # D = Length or Freq
        D = spec.shape[self.dim]
        T = self.mask_width_range[1]
        num_mask = self.num_mask


        if self.dim == 1 & self.adaptive :
          if self.adaptive_number_ratio > 0:
            num_mask = min(int(self.adaptive_number_ratio * D), self.max_n_time_masks)
          if self.adaptive_size_ratio > 0:
            T = min(self.mask_width_range[1], int(self.adaptive_size_ratio * D))

        # mask_length: (B, num_mask, 1)
        mask_length = torch.randint(
            self.mask_width_range[0],
            T,
            (B, num_mask),
            device=spec.device,
        ).unsqueeze(2)

        # mask_pos: (B, num_mask, 1)
        mask_pos = torch.randint(
            0, max(1, D - mask_length.max()), (B, num_mask), device=spec.device
        ).unsqueeze(2)

        # aran: (1, 1, D)
        aran = torch.arange(D, device=spec.device)[None, None, :]
        # mask: (Batch, num_mask, D)
        mask = (mask_pos <= aran) * (aran < (mask_pos + mask_length))
        # Multiply masks: (Batch, num_mask, D) -> (Batch, D)
        mask = mask.any(dim=1)
        if self.dim == 1:
            # mask: (Batch, Length, 1)
            mask = mask.unsqueeze(2)
        elif self.dim == 2:
            # mask: (Batch, 1, Freq)
            mask = mask.unsqueeze(1)

        if self.replace_with_zero:
            value = 0.0
        else:
            value = spec.mean()

        if spec.requires_grad:
            spec = spec.masked_fill(mask, value)
        else:
            spec = spec.masked_fill_(mask, value)
        spec = spec.view(*org_size)
        return spec, spec_lengths

    def forward(self, spec, spec_lengths=None):
        """Forward function.
        Args:
            spec: (Batch, Length, Freq)
        """

        return self.mask_along_axis(
            spec,
            spec_lengths,
        )

