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


class AdaMAML(GBML):
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

        alpha = kwargs.get("alpha", 1.0)
        defaults = dict(betas=betas, eps=eps, correct_bias=True)

        # Customized compute_update for weight_decay
        adam_transform = functools.partial(AdamTransform, alpha=alpha, defaults=defaults)
        compute_update = WeightDecayParameterUpdate(
            parameters=module.parameters(),
            transform=ModuleTransform(adam_transform),
            weight_decay=weight_decay,
        )
        super().__init__(
            module,
            adam_transform,
            lr=lr,
            adapt_transform=adapt_transform,
            first_order=first_order,
            allow_unused=allow_unused,
            allow_nograd=allow_nograd,
            compute_update=compute_update,
            **kwargs
        )

    def clone(
        self,
        first_order=None,
        allow_unused=None,
        allow_nograd=None,
        adapt_transform=None,
    ):
        """
        Return a GBML instance w/ all other behavior the same as a cloned AdaMAML.
        """
        return super().clone(
            adapt_transform=adapt_transform,
            first_order=first_order,
            allow_unused=allow_unused,
            allow_nograd=allow_nograd,
        )

class WeightDecayParameterUpdate(ParameterUpdate):
    def __init__(self, parameters, transform, weight_decay: float = 0.0):
        """
        Arguments:
            parameters (list): Parameters of the model to update.
            transform (Callable): A callable that returns an instantiated transform given a parameter.
            weight_decay (float): Weight decay factor for AdamW.

        Modules:
            transforms_modules (ModuleList): List of transforms per parameter

        Attributes:
            transforms_indeces (list): List of transform ids (or None for non-transforms).
        """
        super().__init__(parameters=parameters, transform=transform)
        self.weight_decay = weight_decay

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
        updates = super().forward(
            loss, parameters,
            create_graph=create_graph,
            retain_graph=retain_graph,
            allow_unused=allow_unused,
            allow_nograd=allow_nograd,
        )

        wd_updates = []
        for p, g, t in zip(list(parameters), updates, self.transforms_indices):
            if t is None or g is None:
                wd_update = g
            else:
                wd_update = g + p * self.weight_decay
            wd_updates.append(wd_update)

        return wd_updates



class AdamTransform(Scale):
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
        self.betas = torch.nn.Parameter(torch.tensor(defaults["betas"]))
        # self.state = {
        #     "step": 0,
        #     "exp_avg": torch.zeros(*shape),
        #     "exp_avg_sq": torch.zeros(*shape),
        # }
        self.step = 0
        # self.exp_avg = self.register_buffer("exp_avg", torch.zeros(shape[0]))
        # self.exp_avg_sq = torch.zeros(shape[0])
        self.register_buffer("exp_avg", torch.zeros(shape[0]))
        self.register_buffer("exp_avg_sq", torch.zeros(shape[0]))

    def forward(self, grad):
        exp_avg, exp_avg_sq = self.exp_avg, self.exp_avg_sq
        beta1, beta2 = self.betas[0], self.betas[1]

        step = self.step + 1

        # exp_avg = torch.add(torch.mul(exp_avg, beta1), grad, alpha=1.0 - beta1)
        # exp_avg_sq = torch.addcmul(torch.mul(exp_avg_sq, beta2), grad, grad, value=1.0 - beta2)
        # denom = exp_avg_sq.sqrt().add_(group["eps"])
        exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
        exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad
        denom = torch.sqrt(exp_avg_sq) + self.defaults["eps"]

        step_size = 1
        if self.defaults["correct_bias"]:  # No bias correction for Bert
            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step
            step_size = step_size * torch.sqrt(bias_correction2) / bias_correction1

        # p.data.addcdiv_(exp_avg, denom, value=-step_size)
        update = grad + step_size * exp_avg / denom

        self.exp_avg = exp_avg
        self.exp_avg_sq = exp_avg_sq
        self.step = step

        return update * self.alpha


# class AdamW(Optimizer):
#     """
#     Implements Adam algorithm with weight decay fix as introduced in
#     `Decoupled Weight Decay Regularization <https://arxiv.org/abs/1711.05101>`__.
#     Parameters:
#         params (:obj:`Iterable[torch.nn.parameter.Parameter]`):
#             Iterable of parameters to optimize or dictionaries defining parameter groups.
#         lr (:obj:`float`, `optional`, defaults to 1e-3):
#             The learning rate to use.
#         betas (:obj:`Tuple[float,float]`, `optional`, defaults to (0.9, 0.999)):
#             Adam's betas parameters (b1, b2).
#         eps (:obj:`float`, `optional`, defaults to 1e-6):
#             Adam's epsilon for numerical stability.
#         weight_decay (:obj:`float`, `optional`, defaults to 0):
#             Decoupled weight decay to apply.
#         correct_bias (:obj:`bool`, `optional`, defaults to `True`):
#             Whether ot not to correct bias in Adam (for instance, in Bert TF repository they use :obj:`False`).
#     """
#     def __init__(
#         self,
#         params: Iterable[torch.nn.parameter.Parameter],
#         lr: float = 1e-3,
#         betas: Tuple[float, float] = (0.9, 0.999),
#         eps: float = 1e-7,
#         weight_decay: float = 0.0,
#         correct_bias: bool = True,
#     ):
#         if lr < 0.0:
#             raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
#         if not 0.0 <= betas[0] < 1.0:
#             raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
#         if not 0.0 <= betas[1] < 1.0:
#             raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
#         if not 0.0 <= eps:
#             raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
#         defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
#         super().__init__(params, defaults)

#     def step(self, closure: Callable = None):
#         """
#         Performs a single optimization step.
#         Arguments:
#             closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
#         """
#         loss = None
#         if closure is not None:
#             loss = closure()

#         for group in self.param_groups:
#             for p in group["params"]:
#                 if p.grad is None:
#                     continue
#                 grad = p.grad.data
#                 if grad.is_sparse:
#                     raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

#                 state = self.state[p]

#                 # State initialization
#                 if len(state) == 0:
#                     state["step"] = 0
#                     # Exponential moving average of gradient values
#                     state["exp_avg"] = torch.zeros_like(p.data)
#                     # Exponential moving average of squared gradient values
#                     state["exp_avg_sq"] = torch.zeros_like(p.data)

#                 exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
#                 beta1, beta2 = group["betas"]

#                 state["step"] += 1

#                 # Decay the first and second moment running average coefficient
#                 # In-place operations to update the averages at the same time
#                 exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
#                 exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
#                 denom = exp_avg_sq.sqrt().add_(group["eps"])

#                 step_size = group["lr"]
#                 if group["correct_bias"]:  # No bias correction for Bert
#                     bias_correction1 = 1.0 - beta1 ** state["step"]
#                     bias_correction2 = 1.0 - beta2 ** state["step"]
#                     step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

#                 p.data.addcdiv_(exp_avg, denom, value=-step_size)

#                 # Just adding the square of the weights to the loss function is *not*
#                 # the correct way of using L2 regularization/weight decay with Adam,
#                 # since that will interact with the m and v parameters in strange ways.
#                 #
#                 # Instead we want to decay the weights in a manner that doesn't interact
#                 # with the m/v parameters. This is equivalent to adding the square
#                 # of the weights to the loss with plain (non-momentum) SGD.
#                 # Add weight decay at the end (fixed version)
#                 if group["weight_decay"] > 0.0:
#                     p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

#         return loss
        
#     def get_lr(self):
#         lr = []
#         for group in self.param_groups:
#             for p in group['params']:
#                 state = self.state[p]
#                 if len(state) == 0:
#                     pass
#                 else:
#                     lr.append(group['lr'])
#         return lr