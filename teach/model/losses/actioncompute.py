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

from typing import Optional

import hydra
import torch
from torch import Tensor
from torch.distributions.distribution import Distribution
from torchmetrics import Metric


class ActionComputeLosses(Metric):
    def __init__(self, vae: bool,
                 dist_sync_on_step=False, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # Save parameters
        self.vae = vae
        losses = ["recons_text2jfeats", "recons_jfeats2jfeats"]
        losses.append("latent_manifold")

        if vae:
            kl_losses = []
            kl_losses.extend(["kl_texts", "kl_motion"])
            losses.extend(kl_losses)

        losses.append("total")

        for loss in losses:
            self.add_state(loss, default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.losses = losses

        # Instantiate loss functions
        self._losses_func = {loss: hydra.utils.instantiate(kwargs[loss + "_func"])
                             for loss in losses if loss != "total"}
        # Save the lambda parameters
        self._params = {loss: kwargs[loss] for loss in losses if loss != "total"}

    def update(self, ds_text, ds_motion, ds_ref,
               lats_text, lat_motion, diss_text, dis_motion, dis_ref, diss_ref):
        total: float = 0.0

        total += self._update_loss("recons_text2jfeats", ds_text.jfeats, ds_ref.jfeats)
        total += self._update_loss("recons_jfeats2jfeats", ds_motion.jfeats, ds_ref.jfeats)

        if self.vae:
            total += self._update_loss("kl_texts", diss_text, diss_ref)
            total += self._update_loss("kl_motion", dis_motion, dis_ref)

        # Average over the actions
        lat_text = torch.stack([action_lat.mean(0) for action_lat in lats_text])
        total += self._update_loss("latent_manifold", lat_text, lat_motion)

        self.total += total.detach()
        self.count += 1

        return total

    def compute(self, split):
        count = getattr(self, "count")
        return {loss: getattr(self, loss)/count for loss in self.losses}

    def _update_loss(self, loss: str, outputs, inputs):
        # Update the loss
        val = self._losses_func[loss](outputs, inputs)
        getattr(self, loss).__iadd__(val.detach())
        # Return a weighted sum
        return self._params[loss] * val

    def loss2logname(self, loss: str, split: str):
        if loss == "total":
            log_name = f"{loss}/{split}"
        else:
            loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{name}/{split}"
        return log_name
