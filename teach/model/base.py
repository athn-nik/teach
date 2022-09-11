import numpy as np
from pytorch_lightning import LightningModule
from hydra.utils import instantiate

from teach.model.metrics import ComputeMetrics
from torchmetrics import MetricCollection
import torch

class BaseModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False)

        # Save visuals, one validation step per validation epoch
        self.store_examples = {"train": None,
                               "val": None,
                               "test": None}

        # Need to define:
        # forward
        # allsplit_step()
        # metrics()
        # losses()

    def __post_init__(self):
        trainable, nontrainable = 0, 0
        for p in self.parameters():
            if p.requires_grad:
                trainable += np.prod(p.size())
            else:
                nontrainable += np.prod(p.size())
        self.hparams.n_params_trainable = trainable
        self.hparams.n_params_nontrainable = nontrainable


    def training_step(self, batch, batch_idx):
        return self.allsplit_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.allsplit_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.allsplit_step("test", batch, batch_idx)

    def allsplit_epoch_end(self, split: str, outputs):
        losses = self.losses[split]
        loss_dict = losses.compute(split)
        dico = {losses.loss2logname(loss, split): value.item()
                for loss, value in loss_dict.items()}
        if split == "val":
            if getattr(self, "metrics_0", None):
                metrics_dict_0 = self.metrics_0.compute()
                metrics_dict_1 = self.metrics_1.compute()
                metrics_dict = {}                
                metrics_dict.update({key+"_0": val for key, val in metrics_dict_0.items()})
                metrics_dict.update({key+"_1": val for key, val in metrics_dict_1.items()})
            else:
                metrics_dict = self.metrics.compute()
            dico.update({f"Metrics/{metric}": value for metric, value in metrics_dict.items()})
        dico.update({"epoch": float(self.trainer.current_epoch),
                     "step": float(self.trainer.current_epoch)})
        # workaround for LR, assuming 1 optimizer, 1 scheduler, very weak
        curr_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        dico.update({'Learning Rate': curr_lr})
        self.log_dict(dico)

    def training_epoch_end(self, outputs):
        return self.allsplit_epoch_end("train", outputs)

    def validation_epoch_end(self, outputs):
        return self.allsplit_epoch_end("val", outputs)

    def test_epoch_end(self, outputs):
        return self.allsplit_epoch_end("test", outputs)

    def configure_optimizers(self):
        optim_dict = {}
        optimizer = instantiate(self.hparams.optim, params=self.parameters())
        optim_dict['optimizer'] = optimizer

        if self.hparams.lr_scheduler == 'reduceonplateau':
            optim_dict['lr_scheduler'] = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1e-3)
            optim_dict['monitor'] = 'losses/total/train'
        elif self.hparams.lr_scheduler == 'steplr':
            optim_dict['lr_scheduler'] = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100)

        return optim_dict 