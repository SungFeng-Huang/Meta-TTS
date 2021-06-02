import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR


def get_scheduler(optimizer, train_config):
    n_warmup_steps = train_config["optimizer"]["warm_up_step"]
    anneal_steps = train_config["optimizer"]["anneal_steps"]
    anneal_rate = train_config["optimizer"]["anneal_rate"]

    def lr_lambda(step):
        """ For lightning with LambdaLR scheduler """
        current_step = step + 1
        lr = np.min(
            [
                np.power(current_step, -0.5),
                np.power(n_warmup_steps, -1.5) * current_step,
            ]
        )
        for s in anneal_steps:
            if current_step > s:
                lr = lr * anneal_rate
        return lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lr_lambda,
    )
    return scheduler
