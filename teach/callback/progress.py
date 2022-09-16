# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import logging

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
import psutil

logger = logging.getLogger(__name__)


class ProgressLogger(Callback):
    def __init__(self,
                 metric_monitor: dict,
                 precision: int = 3):
        # Metric to monitor
        self.metric_monitor = metric_monitor
        self.precision = precision

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule, **kwargs) -> None:
        logger.info("Training started")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule, **kwargs) -> None:
        logger.info("Training done")

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule, **kwargs) -> None:
        if trainer.sanity_checking:
            logger.info("Sanity checking ok.")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, **kwargs) -> None:
        metric_format = f"{{:.{self.precision}e}}"
        line = f"Epoch {trainer.current_epoch}"
        line = f"{line:>{len('Epoch xxxx')}}"  # Right padding
        metrics_str = []

        losses_dict = trainer.callback_metrics
        for metric_name, dico_name in self.metric_monitor.items():
            if dico_name not in losses_dict:
                dico_name = f"losses/{dico_name}"

            if dico_name in losses_dict:
                metric = losses_dict[dico_name].item()
                metric = metric_format.format(metric)
                metric = f"{metric_name} {metric}"
                metrics_str.append(metric)

        if len(metrics_str) == 0:
            return

        memory = f"Memory {psutil.virtual_memory().percent}%"
        line = line + ": " + "   ".join(metrics_str) + "   " + memory
        logger.info(line)
