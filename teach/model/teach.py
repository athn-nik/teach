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

from typing import List, Optional, Union
from torch import Tensor
from torch.distributions.distribution import Distribution

import torch
import numpy as np
from hydra.utils import instantiate

from torch import Tensor
from omegaconf import DictConfig
from teach.model.utils.tools import remove_padding

from teach.model.metrics import ComputeMetricsTeach
from torchmetrics import MetricCollection
from teach.model.base import BaseModel


class TEACH(BaseModel):
    def __init__(self, textencoder: DictConfig,
                 motionencoder: DictConfig,
                 motiondecoder: DictConfig,
                 losses: DictConfig,
                 optim: DictConfig,
                 transforms: DictConfig,
                 nfeats: int,
                 vae: bool,
                 latent_dim: int,
                 motion_branch: bool,
                 hist_frames: int,
                 z_prev_text: Optional[bool] = False,
                 nvids_to_save: Optional[int] = None,
                 teacher_forcing: Optional[bool] = False,
                 **kwargs):
        super().__init__()
        self.textencoder = instantiate(textencoder, nfeats=nfeats)
        if motion_branch:    
            self.motionencoder = instantiate(motionencoder, nfeats=nfeats)
        
        self.transforms = instantiate(transforms)
        self.Datastruct = self.transforms.Datastruct

        self.motiondecoder = instantiate(motiondecoder, nfeats=nfeats)

        self._losses = MetricCollection({split: instantiate(losses, vae=vae,
                                                            motion_branch=motion_branch,
                                                            _recursive_=False)
                                         for split in ["losses_train", "losses_test", "losses_val"]})
        self.losses = {key: self._losses["losses_" + key] for key in ["train", "test", "val"]}
        for k, v in self.store_examples.items():
            self.store_examples[k] = {'text': [], 'keyid': [], 'motions':[]}

        if not self.hparams.losses.loss_on_transition and self.motiondecoder.hparams.prev_data_mode == "hist_frame_outpast":
            raise NotImplementedError

        if self.hparams.losses.loss_on_transition:
            self.metrics = ComputeMetricsTeach()
        else:
            self.metrics_0 = ComputeMetricsTeach()
            self.metrics_1 = ComputeMetricsTeach()

        self.nvids_to_save = nvids_to_save
        # If we want to overide it at testing time
        self.sample_mean = False
        self.fact = 1.0
        self.teacher_forcing = teacher_forcing
        self.hist_frames = hist_frames
        self.previous_latent = False
        if z_prev_text:
            self.previous_latent = z_prev_text
        self.motion_branch = motion_branch
        self.__post_init__()

    def forward_seq(self, texts: list[str], lengths: list[int], align_full_bodies=True, align_only_trans=False,
                    slerp_window_size=None, return_type="joints") -> List[Tensor]:

        assert not (align_full_bodies and align_only_trans)
        do_slerp = slerp_window_size is not None
        # MUST INCLUDE Z PREV
        hframes = None
        prev_z = None
        all_features = []

        for index, (text, length) in enumerate(zip(texts, lengths)):
            current_z, _ = self.encode_data([text], hframes=hframes, z_previous=prev_z, return_latent=True)
            if do_slerp and index > 0:
                length = length - slerp_window_size
                assert length > 1

            current_features = self.motiondecoder(current_z, lengths=[length])[0]
            # create space for slerping
            if do_slerp and index > 0:
                toslerp_inter = torch.tile(0*current_features[0], (slerp_window_size, 1))
                current_features = torch.cat((toslerp_inter, current_features))

            all_features.append(current_features)
            if self.hist_frames > 0:
                hframes = current_features[-self.hist_frames:][None]
            if self.previous_latent:
                prev_z = current_z
        all_features = torch.cat(all_features)
        datastruct = self.Datastruct(features=all_features)

        motion = datastruct.rots
        rots, transl = motion.rots, motion.trans
        pose_rep = "matrix"
        from teach.tools.interpolation import aligining_bodies, slerp_poses, slerp_translation, align_trajectory

        # Rotate bodies etc in place
        end_first_motion = lengths[0] - 1
        for length in lengths[1:]:
            # Compute indices
            begin_second_motion = end_first_motion + 1
            begin_second_motion += slerp_window_size if do_slerp else 0
            # last motion + 1 / to be used with slice
            last_second_motion_ex = end_first_motion + 1 + length

            if align_full_bodies:
                outputs = aligining_bodies(last_pose=rots[end_first_motion],
                                           last_trans=transl[end_first_motion],
                                           poses=rots[begin_second_motion:last_second_motion_ex],
                                           transl=transl[begin_second_motion:last_second_motion_ex],
                                           pose_rep=pose_rep)
                # Alignement
                rots[begin_second_motion:last_second_motion_ex] = outputs[0]
                transl[begin_second_motion:last_second_motion_ex] = outputs[1]
            elif align_only_trans:
                transl[begin_second_motion:last_second_motion_ex] = align_trajectory(transl[end_first_motion],
                                                                                     transl[begin_second_motion:last_second_motion_ex])
            else:
                pass

            # Slerp if needed
            if do_slerp:
                inter_pose = slerp_poses(last_pose=rots[end_first_motion],
                                         new_pose=rots[begin_second_motion],
                                         number_of_frames=slerp_window_size, pose_rep=pose_rep)

                inter_transl = slerp_translation(transl[end_first_motion], transl[begin_second_motion], number_of_frames=slerp_window_size)

                # Fill the gap
                rots[end_first_motion+1:begin_second_motion] = inter_pose
                transl[end_first_motion+1:begin_second_motion] = inter_transl

            # Update end_first_motion
            end_first_motion += length
        from teach.transforms.smpl import RotTransDatastruct
        final_datastruct = self.Datastruct(rots_=RotTransDatastruct(rots=rots, trans=transl))

        if return_type == "vertices":
            return final_datastruct.vertices
        elif return_type == "smpl":
            return { 'rots': rots, 'transl': transl,
                     'vertices': final_datastruct.vertices}
        elif return_type == "joints":
            return final_datastruct.joints
        else:
            raise NotImplementedError


    def sample_from_distribution(self, distribution: Distribution, *,
                                 fact: Optional[bool] = None,
                                 sample_mean: Optional[bool] = False) -> Tensor:
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:
            return distribution.loc

        # Reparameterization trick
        if fact is None:
            return distribution.rsample()

        # Resclale the eps
        eps = distribution.rsample() - distribution.loc
        latent_vector = distribution.loc + fact * eps
        return latent_vector

    def text_to_motion_forward(self, text_sentences: List[str],
                               lengths: List[int],
                               first_frames: List[Tensor],
                               *, return_latent: bool = False):
        # Encode the text to the latent space
        if self.hparams.vae:
            distribution = self.textencoder(text_sentences)
            latent_vector = self.sample_from_distribution(distribution)
        else:
            distribution = None
            latent_vector = self.textencoder(text_sentences)

        # Decode the latent vector to a motion
        features = self.motiondecoder(latent_vector, first_frames, lengths)
        datastruct = self.Datastruct(features=features)

        if not return_latent:
            return datastruct
        return datastruct, latent_vector, distribution

    def motion_to_motion_forward(self, datastruct,
                                 lengths: Optional[List[int]],
                                 first_frames: List[Tensor],
                                 *, return_latent: bool = False):
        # Make sure it is on the good device
        datastruct.transforms = self.transforms

        # Encode the motion to the latent space
        if self.hparams.vae:
            distribution = self.motionencoder(datastruct.features, lengths)
            latent_vector = self.sample_from_distribution(distribution)
        else:
            distribution = None
            latent_vector: Tensor = self.motionencoder(datastruct.features, lengths)

        # Decode the latent vector to a motion
        features = self.motiondecoder(latent_vector, first_frames, lengths)
        datastruct = self.Datastruct(features=features)

        if not return_latent:
            return datastruct
        return datastruct, latent_vector, distribution


    def encode_data(self, data: Union[List[str], Tensor],
                    hframes=None,
                    z_previous=None,
                    *, return_latent: bool = False):
        # Use the text branch
        # and encode the text to the latent space
        if isinstance(data[0], str):
            encoded = self.textencoder(data, hframes=hframes, z_pt=z_previous)
        else: 
            # it is a motion, and we don't care about the past frames
            assert hframes is None
            assert z_previous is None
            encoded = self.motionencoder(data)
        
        if self.hparams.vae:
            distribution = encoded
            latent_vector = self.sample_from_distribution(distribution)
        else:
            distribution = None
            latent_vector = encoded

        if not return_latent:
            return distribution
        return latent_vector, distribution

    def allsplit_step(self, split: str, batch, batch_idx):

        # Prepare the generated motion features
        length_0, length_1 = batch["length_0"], batch["length_1"]
        length_transition = batch["length_transition"]
        length_1_with_transition = batch["length_1_with_transition"]

        total_length = [x+y+z for x, y, z in zip(length_0, length_transition, length_1)]

        # input_motion_feats_0: [batch_size, max_time, feats]

        input_motion_feats_0, input_motion_feats_1 = batch["motion_feats_0"], batch["motion_feats_1"]
        input_motion_feats_1_with_transition = batch["motion_feats_1_with_transition"]

        # Prepare the groundtruth motion features
        input_motion_feats_0_lst = [feats[:len0] for feats, len0 in zip(input_motion_feats_0, length_0)]
        if self.hparams.losses.loss_on_transition:
            input_motion_feats_1_with_transition_lst = [feats[:(len1+trans_len)] for feats, trans_len, len1 in zip(input_motion_feats_1_with_transition, length_transition, length_1)]
        else:
            input_motion_feats_1_lst = [feats[trans_len:][:len1] for feats, trans_len, len1 in zip(input_motion_feats_1_with_transition, length_transition, length_1)]


        ### TEXT PART
        text_0, text_1 = batch["text_0"], batch["text_1"]
        # breakpoint()
        #### Encode the first text
        latent_vector_0_T, distribution_0_T = self.encode_data(text_0, return_latent=True)

        #### Decode into motions
        ## First motion pass
        # Decode the latent vector to a motion
        # bs, hist_frames+length_0
        output_features_0_T = self.motiondecoder(latent_vector_0_T, lengths=length_0)
        output_features_0_T_lst = [feats[:len0] for feats, len0 in zip(output_features_0_T, length_0)]


        ## Second pass
        # compute hist_frames
        # hist_frames
        if self.hist_frames > 0:
            # TODO: create config for teacher forcing            
            if self.teacher_forcing:                
                hist_frames_tensor = torch.stack([x[-self.hist_frames:] for x in input_motion_feats_0_lst])
            else:            
                hist_frames_tensor = torch.stack([x[-self.hist_frames:] for x in output_features_0_T_lst])        
        else:
            hist_frames_tensor = None

        if self.previous_latent:
            latent_vector_1_T, distribution_1_T = self.encode_data(text_1, hframes=hist_frames_tensor,
                                                                   z_previous=latent_vector_0_T, return_latent=True)
        else:
            latent_vector_1_T, distribution_1_T = self.encode_data(text_1, hframes=hist_frames_tensor, return_latent=True)

        output_features_1_T_with_transition = self.motiondecoder(latent_vector_1_T, lengths=length_1_with_transition)

        # to compute the loss
        # need to collate, or
        output_features_1_T_lst = [feats[trans_len:][:len1] for feats, trans_len, len1 in zip(output_features_1_T_with_transition, length_transition, length_1)]

        if self.hparams.losses.loss_on_transition:
            output_features_1_T_with_transition_lst = [feats[:(len1+trans_len)] for feats, trans_len, len1 in zip(output_features_1_T_with_transition, length_transition, length_1)]
        output_features_0_M_lst = None
        output_features_0_M = None
        latent_vector_0_M = None
        distribution_0_M = None
        output_features_1_M_with_transition = None
        output_features_1_M_with_transition = None
        output_features_1_M_lst = None
        output_features_1_M_with_transition_lst = None
        distribution_1_M = None
        distribution_0_M = None
        latent_vector_1_M = None
        ## Motion part
        if self.motion_branch:
            #### Encode the first motion
            latent_vector_0_M, distribution_0_M = self.encode_data(input_motion_feats_0, return_latent=True)
            output_features_0_M = self.motiondecoder(latent_vector_0_M, lengths=length_0)
            output_features_0_M_lst = [feats[:len0] for feats, len0 in zip(output_features_0_M, length_0)]

            latent_vector_1_M, distribution_1_M = self.encode_data(input_motion_feats_1_with_transition, return_latent=True)

            output_features_1_M_with_transition = self.motiondecoder(latent_vector_1_M, lengths=length_1_with_transition)

            # to compute the loss
            # need to collate, or
            output_features_1_M_lst = [feats[trans_len:][:len1] for feats, trans_len, len1 in zip(output_features_1_M_with_transition, length_transition, length_1)]

            if self.hparams.losses.loss_on_transition:
                output_features_1_M_with_transition_lst = [feats[:(len1+trans_len)] for feats, trans_len, len1 in zip(output_features_1_M_with_transition, length_transition, length_1)]
   
        if self.hparams.vae:
            # Create a centred normal distribution to compare with
            mu_ref = torch.zeros_like(distribution_0_T.loc)
            scale_ref = torch.ones_like(distribution_0_T.scale)
            distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)
        else:
            distribution_ref = None

        # Compute the losses
        if self.hparams.losses.loss_on_transition:
            loss = self.losses[split].update(input_motion_feats_0_lst,
                                             input_motion_feats_1_with_transition_lst,
                                             output_features_0_T_lst,
                                             output_features_1_T_with_transition_lst,
                                             output_features_0_M_lst,
                                             output_features_1_M_with_transition_lst,
                                             distribution_0_T,
                                             distribution_1_T,
                                             distribution_0_M,
                                             distribution_1_M,
                                             distribution_ref,
                                             latent_vector_0_T,
                                             latent_vector_1_T,
                                             latent_vector_0_M,
                                             latent_vector_1_M)
        else:
            loss = self.losses[split].update(input_motion_feats_0_lst,
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
                                            latent_vector_1_M)

        return loss
