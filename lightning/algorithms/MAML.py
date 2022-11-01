from learn2learn.algorithms import GBML
from learn2learn.optim.transforms import ModuleTransform, ReshapedTransform
from learn2learn.nn.misc import Scale, Lambda

from torch.nn import Module, Identity
from typing import Literal
import functools


class MAML(GBML):
    def __init__(
        self,
        module: Module,
        lr: float = 1.0,
        first_order: bool = False,
        allow_unused: bool = False,
        allow_no_grad: bool = False,
        **kwargs,
    ):
        # transform: Union[Callable[[Parameter], Module], Type[Module]],
        transform = ModuleTransform(Identity)
        adapt_transform: bool = False
        super().__init__(module, transform, lr, adapt_transform, first_order,
                         allow_unused, allow_no_grad, **kwargs)


    class ParameterWiseScale(Scale):
        def __init__(self, *shape, alpha=1.0):
            super().__init__((1,), alpha)

    class ElementWiseScale(Scale):
        def __init__(self, *shape, alpha=1.0):
            super().__init__(shape, alpha)


class MetaSGD(GBML):
    def __init__(
        self,
        module: Module,
        lr: float = 1.0,
        first_order: bool = False,
        allow_unused: bool = False,
        allow_no_grad: bool = False,
        **kwargs,
    ):
        # lr for module: lr
        # lr for scale_transform: lr * alpha
        alpha = kwargs.get("alpha", 1.0)
        scale_type = kwargs.get("scale_type", "element")
        scale_transform = {
            "parameter": functools.partial(self.ParameterWiseScale, alpha=alpha),
            "element": functools.partial(self.ElementWiseScale, alpha=alpha),
        }[scale_type]
        transform = ModuleTransform(scale_transform)
        adapt_transform: bool = False   # whether to adapt in inner-loop
        super().__init__(module, transform, lr, adapt_transform, first_order,
                         allow_unused, allow_no_grad, **kwargs)


class AlphaMAML(GBML):
    def __init__(
        self,
        module: Module,
        lr: float = 1.0,
        first_order: bool = False,
        allow_unused: bool = False,
        allow_no_grad: bool = False,
        **kwargs,
    ):
        # lr for module: lr
        # lr for scale_transform: lr * alpha
        alpha = kwargs.get("alpha", 1.0)
        scale_type = kwargs.get("scale_type", "parameter")
        scale_transform = {
            "parameter": functools.partial(self.ParameterWiseScale, alpha=alpha),
            "element": functools.partial(self.ElementWiseScale, alpha=alpha),
        }[scale_type]
        transform = ModuleTransform(scale_transform)
        adapt_transform: bool = False   # whether to adapt in inner-loop
        super().__init__(module, transform, lr, adapt_transform, first_order,
                         allow_unused, allow_no_grad, **kwargs)
