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

import hydra
import torch

from torchmetrics import Metric


class TeachComputeLosses(Metric):
    def __init__(self, vae: bool,
                 mode: str,
                 motion_branch: bool = False,
                 loss_on_both: bool = True,
                 loss_on_jfeats: bool = False,
                 loss_on_velocities: bool= False,
                 dist_sync_on_step=False, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # Save parameters
        self.vae = vae
        self.mode = mode
        self.loss_on_both = loss_on_both
        self.motion_branch = motion_branch
        self.loss_on_jfeats = loss_on_jfeats
        self.loss_on_velocities = loss_on_velocities

        if loss_on_jfeats:
            raise NotImplementedError

        losses = []
        if mode == "xyz" or loss_on_jfeats:
            if motion_branch:
                losses.append("recons_jfeats2jfeats_0")
                losses.append("recons_jfeats2jfeats_1")
            losses.append("recons_text2jfeats_0")
            losses.append("recons_text2jfeats_1")
        if mode == "smpl":
            if motion_branch:
                losses.append("recons_rfeats2rfeats_0")
                losses.append("recons_rfeats2rfeats_1")
            losses.append("recons_text2rfeats_0")
            losses.append("recons_text2rfeats_1")
        else:
            ValueError("This mode is not recognized.")
        
        if loss_on_velocities:
            losses.append('recons_text2rfeats_vel_0')
            losses.append('recons_text2rfeats_vel_1')
            if motion_branch:
                losses.append("recons_rfeats2rfeats_vel_0")
                losses.append("recons_rfeats2rfeats_vel_1")

        if vae or loss_on_both:
            kl_losses = []
            kl_losses.extend(["kl_text2motion_0", "kl_motion2text_0", "kl_text2motion_1", "kl_motion2text_1"])
            kl_losses.extend(["kl_text_0", "kl_text_1"])
            if motion_branch:
                kl_losses.extend(["kl_motion_0", "kl_motion_1"])
            losses.extend(kl_losses)
        if not self.vae or loss_on_both:
            if motion_branch:
                losses.append("latent_manifold_0")
                losses.append("latent_manifold_1")
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

    def update(self, input_motion_feats_0_lst, 
               input_motion_feats_1_lst,
               output_features_0_T_lst,
               output_features_1_T_lst,
               output_features_0_M_lst,
               output_features_1_M_lst,
               distribution_0_T,
               distribution_1_T,
               distribution_0_M,
               distribution_1_M,
               distribution_ref,
               latent_vector_0_T,
               latent_vector_1_T,
               latent_vector_0_M,
               latent_vector_1_M,
               
               ):
        total: float = 0.0            

        if self.mode == "xyz" or self.loss_on_jfeats:
            total += self._update_loss("recons_text2jfeats_0", input_motion_feats_0_lst, output_features_0_T_lst)
            total += self._update_loss("recons_text2jfeats_1", input_motion_feats_1_lst, output_features_1_T_lst)

            if self.motion_branch:
                total += self._update_loss("recons_jfeats2jfeats_0", input_motion_feats_0_lst, output_features_0_M_lst)
                total += self._update_loss("recons_jfeats2jfeats_1", input_motion_feats_1_lst, output_features_1_M_lst)


        if self.mode == "smpl":
            total += self._update_loss("recons_text2rfeats_0", input_motion_feats_0_lst, output_features_0_T_lst)
            total += self._update_loss("recons_text2rfeats_1", input_motion_feats_1_lst, output_features_1_T_lst)

            if self.motion_branch:
                total += self._update_loss("recons_rfeats2rfeats_0", input_motion_feats_0_lst, output_features_0_M_lst)
                total += self._update_loss("recons_rfeats2rfeats_1", input_motion_feats_1_lst, output_features_1_M_lst)

        if self.vae or self.loss_on_both:
            if self.motion_branch:
                total += self._update_loss("kl_text2motion_0", distribution_0_T, distribution_0_M)
                total += self._update_loss("kl_motion2text_0", distribution_0_M, distribution_0_T)
                total += self._update_loss("kl_text2motion_1", distribution_1_T, distribution_1_M)
                total += self._update_loss("kl_motion2text_1", distribution_1_M, distribution_1_T)

            total += self._update_loss("kl_text_0", distribution_0_T, distribution_ref)
            total += self._update_loss("kl_text_1", distribution_1_T, distribution_ref)

            if self.motion_branch:
                total += self._update_loss("kl_motion_0", distribution_0_M, distribution_ref)
                total += self._update_loss("kl_motion_1", distribution_1_M, distribution_ref)

        if not self.vae or self.loss_on_both:
            if self.motion_branch:
                total += self._update_loss("latent_manifold_0", latent_vector_0_T, latent_vector_0_M)
                total += self._update_loss("latent_manifold_1", latent_vector_1_T, latent_vector_1_M)

        if self.loss_on_velocities:
            vel_0_in = [torch.diff(m[..., 3:], dim=0) for m in input_motion_feats_0_lst]
            vel_0_out = [torch.diff(m[..., 3:], dim=0) for m in output_features_0_T_lst]
            vel_1_in = [torch.diff(m[..., 3:], dim=0) for m in input_motion_feats_1_lst]
            vel_1_out = [torch.diff(m[..., 3:], dim=0) for m in output_features_1_T_lst]
            total += self._update_loss("recons_text2rfeatsvel_0", vel_0_in, vel_0_out)
            total += self._update_loss("recons_text2rfeatsvel_1", vel_1_in, vel_1_out )
            
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
        weighted_loss = self._params[loss] * val
        return weighted_loss

    def loss2logname(self, loss: str, split: str):
        if loss == "total":
            log_name = f"losses/{loss}/{split}"
        else:
            if loss.endswith("0") or loss.endswith("1"):
               loss_type, name = loss.split("_")[:2] 
            else:
                loss_type, name = loss.split("_")
            log_name = f"losses/{loss_type}/{name}/{split}"
        return log_name
