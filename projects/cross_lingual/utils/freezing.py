import torch
import torch.nn as nn


def get_norm_params(model: nn.Module):
    res = []
    for module in model.modules():
        if isinstance(module, nn.LayerNorm) or isinstance(module, nn.BatchNorm1d):
            for param in module.parameters():
                res.append(param)
    return res


def get_bias_params(model: nn.Module):
    res = []
    for param in model.named_parameters():
        if ".bias" in param[0]:
            res.append(param[1])
    return res
