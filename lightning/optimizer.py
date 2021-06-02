import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR


def get_optimizer(model, model_config, train_config):
    n_warmup_steps = train_config["optimizer"]["warm_up_step"]
    anneal_steps = train_config["optimizer"]["anneal_steps"]
    anneal_rate = train_config["optimizer"]["anneal_rate"]
    init_lr = np.power(model_config["transformer"]["encoder_hidden"], -0.5)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=init_lr,
        betas=train_config["optimizer"]["betas"],
        eps=train_config["optimizer"]["eps"],
        weight_decay=train_config["optimizer"]["weight_decay"],
    )
    return optimizer
