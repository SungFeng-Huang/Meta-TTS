from learn2learn.algorithms import GBML
from learn2learn.optim.transforms import ModuleTransform, ReshapedTransform
from learn2learn.nn.misc import Scale, Lambda
from learn2learn.optim import ParameterUpdate
from learn2learn.utils import clone_module

import math
import torch
from torch.optim import Optimizer
from torch.nn import Module, Identity
from typing import Literal, Tuple, Iterable, Callable
import functools
import traceback

from lightning.algorithms.utils import GBML


class ParameterWiseScale(Scale):
    """To be compatible with ModuleTransform"""
    def __init__(self, *shape, alpha=1.0):
        super().__init__(None, alpha)

class ElementWiseScale(Scale):
    """To be compatible with ModuleTransform"""
    def __init__(self, *shape, alpha=1.0):
        super().__init__(shape, alpha)


class MAML(GBML):
    def __init__(
        self,
        module: Module,
        lr: float = 1.0,
        first_order: bool = False,
        allow_unused: bool = False,
        allow_nograd: bool = False,
        **kwargs,
    ):
        # transform: Union[Callable[[Parameter], Module], Type[Module]],
        transform = ModuleTransform(Identity)
        adapt_transform: bool = False
        super().__init__(
            module,
            transform,
            lr=lr,
            adapt_transform=adapt_transform,
            first_order=first_order,
            allow_unused=allow_unused,
            allow_nograd=allow_nograd,
            **kwargs
        )


class MetaSGD(GBML):
    def __init__(
        self,
        module: Module,
        lr: float = 1.0,
        first_order: bool = False,
        allow_unused: bool = False,
        allow_nograd: bool = False,
        **kwargs,
    ):
        # lr for module: lr
        # lr for scale_transform: lr * alpha
        alpha = kwargs.get("alpha", 1.0)
        scale_type = kwargs.get("scale_type", "element")
        scale_transform = {
            "parameter": functools.partial(ParameterWiseScale, alpha=alpha),
            "element": functools.partial(ElementWiseScale, alpha=alpha),
        }[scale_type]
        transform = ModuleTransform(scale_transform)
        adapt_transform: bool = False   # whether to adapt in inner-loop
        super().__init__(
            module,
            transform,
            lr=lr,
            adapt_transform=adapt_transform,
            first_order=first_order,
            allow_unused=allow_unused,
            allow_nograd=allow_nograd,
            **kwargs
        )


class AlphaMAML(GBML):
    def __init__(
        self,
        module: Module,
        lr: float = 1.0,
        first_order: bool = False,
        allow_unused: bool = False,
        allow_nograd: bool = False,
        **kwargs,
    ):
        # lr for module: lr
        # lr for scale_transform: lr * alpha
        alpha = kwargs.get("alpha", 1.0)
        scale_type = kwargs.get("scale_type", "parameter")
        scale_transform = {
            "parameter": functools.partial(ParameterWiseScale, alpha=alpha),
            "element": functools.partial(ElementWiseScale, alpha=alpha),
        }[scale_type]
        transform = ModuleTransform(scale_transform)
        adapt_transform: bool = False   # whether to adapt in inner-loop
        super().__init__(
            module,
            transform,
            lr=lr,
            adapt_transform=adapt_transform,
            first_order=first_order,
            allow_unused=allow_unused,
            allow_nograd=allow_nograd,
            **kwargs
        )


class AdamWMAML(GBML):
    def __init__(
        self,
        module,
        lr=1.0,
        adapt_transform=False,
        first_order=False,
        allow_unused=False,
        allow_nograd=False,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        **kwargs
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))

        assert correct_bias
        alpha = kwargs.get("alpha", 1.0)
        defaults = dict(betas=betas, eps=eps, correct_bias=correct_bias)

        # Customized compute_update for weight_decay
        adamw_transform = functools.partial(AdamWTransform, alpha=alpha, defaults=defaults)
        compute_update = AdvancedParameterUpdate(
            parameters=module.parameters(),
            transform=ModuleTransform(adamw_transform),
            weight_decay=weight_decay,
        )
        super().__init__(
            module,
            adamw_transform,
            lr=lr,
            adapt_transform=adapt_transform,
            first_order=first_order,
            allow_unused=allow_unused,
            allow_nograd=allow_nograd,
            compute_update=compute_update,
            **kwargs
        )


class AdvancedParameterUpdate(ParameterUpdate):
    def __init__(
        self,
        parameters,
        transform,
        weight_decay: float = 0.0,
        clip_grad_norm: float = 0.0,
    ):
        """
        Advanced ParameterUpdate with WD and clip_grad_norm available.

        Arguments:
            parameters (list): Parameters of the model to update.
            transform (Callable): A callable that returns an instantiated transform given a parameter.
            weight_decay (float): Weight decay factor for AdamW.
            clip_grad_norm (float): If > 0, perform clip_grad_norm_() before Adam optimization.

        Modules:
            transforms_modules (ModuleList): List of transforms per parameter

        Attributes:
            transforms_indeces (list): List of transform ids (or None for non-transforms).
        """
        super().__init__(parameters=parameters, transform=transform)
        # self.weight_decay = weight_decay
        # self.clip_grad_norm = clip_grad_norm
        self.register_buffer("weight_decay", torch.tensor(weight_decay))
        self.register_buffer("clip_grad_norm", torch.tensor(clip_grad_norm))

    def forward(
        self,
        loss,
        parameters,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
        allow_nograd=False,
    ):
        """
        **Description**

        Similar to torch.autograd.grad, but passes the gradients through the
        provided transform.

        **Arguments**

        * **loss** (Tensor) - The loss to differentiate.
        * **parameters** (iterable) - Parameters w.r.t. which we want to compute the update.
        * **create_graph** (bool, *optional*, default=False) - Same as `torch.autograd.grad`.
        * **retain_graph** (bool, *optional*, default=False) - Same as `torch.autograd.grad`.
        * **allow_unused** (bool, *optional*, default=False) - Same as `torch.autograd.grad`.
        * **allow_nograd** (bool, *optional*, default=False) - Properly handles parameters
            that do not require gradients. (Their update will be `None`.)

        """
        parameters = list(parameters)
        if allow_nograd:
            parameters = list(parameters)
            diff_params = [p for p in parameters if p.requires_grad]
            grad_params = torch.autograd.grad(
                loss,
                diff_params,
                retain_graph=create_graph,
                create_graph=create_graph,
                allow_unused=allow_unused)
            gradients = []

            # Handles gradients for non-differentiable parameters
            grad_counter = 0
            for param in parameters:
                if param.requires_grad:
                    gradient = grad_params[grad_counter]
                    grad_counter += 1
                else:
                    gradient = None
                gradients.append(gradient)
        else:
            try:
                gradients = torch.autograd.grad(
                    loss,
                    parameters,
                    create_graph=create_graph,
                    retain_graph=retain_graph,
                    allow_unused=allow_unused,
                )
            except RuntimeError:
                traceback.print_exc()
                msg = 'learn2learn: Maybe try with allow_nograd=True and/or' +\
                      'allow_unused=True ?'
                print(msg)
                raise

        if self.clip_grad_norm > 0:
            updates = []
            total_norm = torch.norm(torch.stack([torch.norm(g, 2.0) for g in gradients if g is not None]), 2.0)
            clip_coef = self.clip_grad_norm / (total_norm + 1e-6)
            clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
            for g in gradients:
                if g is not None:
                    g = g * clip_coef_clamped
                updates.append(g)
        else:
            updates = gradients

        wd_updates = []
        for p, g, t in zip(list(parameters), updates, self.transforms_indices):
            if g is None:
                wd_update = g
            elif t is None:
                wd_update = g + p * self.weight_decay
            else:
                transform = self.transforms_modules[t]
                update = transform(g)
                wd_update = update + p * self.weight_decay
            wd_updates.append(wd_update)

        return wd_updates



class AdamWTransform(Scale):
    """
    ElementWiseScale per parameter for Adam operation
    """
    def __init__(self, *shape, alpha=1.0, defaults: dict = None):
        """
        Arguments:
            shape: (num_element, num_element)

        Hyper-parameters (constant):
            defaults:
                correct_bias (bool): True
                eps (float): 1e-7

        Parameters (fix in inner loop):
            alpha (float): learning rate ratio (learning rate = lr * alpha)
            betas (Tuple[float, float]): beta1, beta2
        
        Buffers:
            exp_avg (Tensor): start from 0
            exp_avg_sq (Tensor): start from 0

        Attributes:
            step (int): start from 0
        """
        super().__init__(alpha=alpha)
        self.defaults = defaults
        self.betas = torch.nn.Parameter(torch.tensor(defaults["betas"]), requires_grad=False)
        # self.state = {
        #     "step": 0,
        #     "exp_avg": torch.zeros(*shape),
        #     "exp_avg_sq": torch.zeros(*shape),
        # }
        self.register_buffer("step", torch.tensor(0.))
        self.register_buffer("exp_avg", torch.zeros(shape[0]))
        self.register_buffer("exp_avg_sq", torch.zeros(shape[0]))

    def forward(self, grad):
        exp_avg, exp_avg_sq = self.exp_avg, self.exp_avg_sq
        beta1, beta2 = self.betas[0], self.betas[1]

        step = self.step + 1

        exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
        exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad
        denom = torch.sqrt(exp_avg_sq) + self.defaults["eps"]

        step_size = 1
        if self.defaults["correct_bias"]:  # No bias correction for Bert
            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step
            step_size = step_size * torch.sqrt(bias_correction2) / bias_correction1

        update = step_size * exp_avg / denom

        self.exp_avg = exp_avg
        self.exp_avg_sq = exp_avg_sq
        self.step = step

        return update * self.alpha
