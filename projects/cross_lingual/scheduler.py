import numpy as np
from torch.optim.lr_scheduler import LambdaLR


def get_scheduler(optimizer, train_config):
    scheduler_type = train_config.get("scheduler_type", "sqrt")
    if scheduler_type == "sqrt":
        lr_lambda = sqrt_schedule
    elif scheduler_type == "const":
        lr_lambda = const_schedule
    else:
        raise NotImplementedError
        
    scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lr_lambda(train_config),
    )
    return scheduler


def sqrt_schedule(train_config):
    n_warmup_steps = train_config["optimizer"].get("warm_up_step", 0)
    anneal_steps = train_config["optimizer"].get("anneal_steps", [])
    anneal_rate = train_config["optimizer"].get("anneal_rate", 1.0)

    def lr_lambda(step):
        if n_warmup_steps > 0:
            current_step = step + 1
            if current_step <= n_warmup_steps:
                factor = current_step / n_warmup_steps
            else:
                factor = np.power(n_warmup_steps / current_step, 0.5)
        else:
            factor = 1
        for s in anneal_steps:
            if current_step > s:
                factor = factor * anneal_rate
        return factor
    
    return lr_lambda


def const_schedule(train_config):
    n_warmup_steps = train_config["optimizer"].get("warm_up_step", 0)
    anneal_steps = train_config["optimizer"].get("anneal_steps", [])
    anneal_rate = train_config["optimizer"].get("anneal_rate", 1.0)

    def lr_lambda(step):
        if n_warmup_steps > 0:
            current_step = step + 1
            if current_step <= n_warmup_steps:
                factor = current_step / n_warmup_steps
            else:
                factor = 1
        else:
            factor = 1
        for s in anneal_steps:
            if current_step > s:
                factor = factor * anneal_rate
        return factor
    
    return lr_lambda
