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

import pytorch_lightning as pl
import torch
from torch import nn

from teach.data.tools import PoseData


class Dummy(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.store_examples = {"train": None,
                               "val": None,
                               "test": None}

    def forward(self, batch: dict) -> PoseData:
        return batch["pose_data"]

    def allsplit_step(self, split: str, batch, batch_idx):
        joints = batch["pose_data"].joints
        if batch_idx == 0:
            self.store_examples[split] = {
                "text": batch["text"],
                "length": batch["length"],
                "ref": joints,
                "from_text": joints,
                "from_motion": joints
            }

        x = batch["pose_data"].poses
        x = torch.rand((5, 10), device=x.device)
        xhat = self.linear(x)

        loss = torch.nn.functional.mse_loss(x, xhat)
        return loss

    def training_step(self, batch, batch_idx):
        return self.allsplit_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self.allsplit_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self.allsplit_step("test", batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(lr=1e-4, params=self.parameters())
        return {"optimizer": optimizer}