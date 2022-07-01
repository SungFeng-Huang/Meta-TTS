from .base import BaseAdaptorSystem
from .fit import BaseAdaptorFitSystem
from .validate import BaseAdaptorValidateSystem
from .test import BaseAdaptorTestSystem
from lightning.utils import loss2dict


class BaselineFitSystem(BaseAdaptorFitSystem):

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        """ Normal forwarding.

        Function:
            common_step(): Defined in `lightning.systems.system.System`
        """
        ref_psd = (batch["p_targets"].unsqueeze(2)
                   if self.reference_prosody else None)
        output = self(**batch, reference_prosody=ref_psd)
        loss = self.loss_func(batch, output)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}":v for k,v in loss2dict(loss).items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=len(batch["ids"]))
        return {
            'loss': loss[0],
            'losses': [l.detach() for l in loss],
            'output': [o.detach() for o in output],
        }

BaselineValidateSystem = BaseAdaptorValidateSystem
BaselineTestSystem = BaseAdaptorTestSystem
