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

from typing import List, Optional

import torch
import numpy as np
from hydra.utils import instantiate

from torch import Tensor
from omegaconf import DictConfig
from teach.model.utils.tools import remove_padding

from teach.model.metrics import ComputeMetrics
from torchmetrics import MetricCollection
from teach.model.base import BaseModel


class TEMOS(BaseModel):
    def __init__(self, textencoder: DictConfig,
                 motionencoder: DictConfig,
                 motiondecoder: DictConfig,
                 losses: DictConfig,
                 optim: DictConfig,
                 transforms: DictConfig,
                 nfeats: int,
                 vae: bool,
                 latent_dim: int,
                 motion_branch: Optional[bool] = False,
                 nvids_to_save: Optional[int] = None,
                 **kwargs):
        super().__init__()
        self.textencoder = instantiate(textencoder)

        self.motionencoder = instantiate(motionencoder, nfeats=nfeats)

        self.transforms = instantiate(transforms)
        self.Datastruct = self.transforms.Datastruct

        self.motiondecoder = instantiate(motiondecoder, nfeats=nfeats)
        self.motion_branch = motion_branch
        self._losses = MetricCollection({split: instantiate(losses, vae=vae,
                                                            motion_branch=motion_branch,
                                                            _recursive_=False)
                                         for split in ["losses_train", "losses_test", "losses_val"]})
        self.losses = {key: self._losses["losses_" + key] for key in ["train", "test", "val"]}
        self.metrics = ComputeMetrics()
        self.nvids_to_save = nvids_to_save
        # If we want to overide it at testing time
        self.sample_mean = False
        self.fact = 1.0
        for k, v in self.store_examples.items():
            self.store_examples[k] = {'text': [], 'keyid': [], 'motions':[]}

        self.__post_init__()

    # Forward: text => motion
    def forward(self, batch: dict, return_rots=False) -> List[Tensor]:
        datastruct_from_text = self.text_to_motion_forward(batch["text"],
                                                           batch["length"])
        if return_rots:
            return remove_padding(datastruct_from_text.rots.rots, batch["length"]), remove_padding(datastruct_from_text.rots.trans, batch["length"])

        return remove_padding(datastruct_from_text.joints, batch["length"])

    def forward_seq(self, texts: list[str], lengths: list[int], align_full_bodies=True, align_only_trans=False,
                    slerp_window_size=None, return_type="joints") -> List[Tensor]:

        assert not (align_full_bodies and align_only_trans)
        do_slerp = slerp_window_size is not None

        all_features = []
        for index, (text, length) in enumerate(zip(texts, lengths)):
            # create space for slerping
            if do_slerp and index > 0:
                length = length - slerp_window_size
                assert length > 1

            current_features = self.text_to_motion_forward([text], [length]).features[0]

            if do_slerp and index > 0:
                toslerp_inter = torch.tile(0*current_features[0], (slerp_window_size, 1))
                current_features = torch.cat((toslerp_inter, current_features))
            all_features.append(current_features)
        
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
        elif return_type in ["joints", 'mmmns', 'mmm']:
            return final_datastruct.joints
        else:
            raise NotImplementedError

    def text_to_motion_forward(self, text_sentences: List[str], lengths: List[int],
                               return_latent: bool = False,
                               return_feats: bool = False,
                               ):
        # Encode the text to the latent space
        if self.hparams.vae:
            distribution = self.textencoder(text_sentences)

            if self.sample_mean:
                latent_vector = distribution.loc
            else:
                # Reparameterization trick
                eps = distribution.rsample() - distribution.loc
                latent_vector = distribution.loc + self.fact * eps
        else:
            distribution = None
            latent_vector = self.textencoder(text_sentences)

        # Decode the latent vector to a motion
        features = self.motiondecoder(latent_vector, lengths)
        datastruct = self.Datastruct(features=features)

        if not return_latent:
            if return_feats:
                return features
            else:
                return datastruct
        if return_feats:
            return features, latent_vector, distribution
        else:
            return datastruct, latent_vector, distribution

    def motion_to_motion_forward(self, datastruct,
                                 lengths: Optional[List[int]] = None,
                                 return_latent: bool = False
                                 ):
        # Make sure it is on the good device
        datastruct.transforms = self.transforms

        # Encode the motion to the latent space
        if self.hparams.vae:
            distribution = self.motionencoder(datastruct.features, lengths)

            if self.sample_mean:
                latent_vector = distribution.loc
            else:
                # Reparameterization trick
                eps = distribution.rsample() - distribution.loc
                latent_vector = distribution.loc + self.fact * eps
        else:
            distribution = None
            latent_vector: Tensor = self.motionencoder(datastruct.features, lengths)

        # Decode the latent vector to a motion
        features = self.motiondecoder(latent_vector, lengths)
        datastruct = self.Datastruct(features=features)

        if not return_latent:
            return datastruct
        return datastruct, latent_vector, distribution

    def allsplit_step(self, split: str, batch, batch_idx):
        # Encode the text/decode to a motion
        ret = self.text_to_motion_forward(batch["text"],
                                          batch["length"],
                                          return_latent=True)
        datastruct_from_text, latent_from_text, distribution_from_text = ret
        if self.motion_branch:

            # Encode the motion/decode to a motion
            ret = self.motion_to_motion_forward(batch["datastruct"],
                                                batch["length"],
                                                return_latent=True)
            datastruct_from_motion, latent_from_motion, distribution_from_motion = ret
        else:
            datastruct_from_motion = None
            latent_from_motion = None
            distribution_from_motion = None
        # GT data
        datastruct_ref = batch["datastruct"]

        # Compare to a Normal distribution
        if self.hparams.vae:
            # Create a centred normal distribution to compare with
            mu_ref = torch.zeros_like(distribution_from_text.loc)
            scale_ref = torch.ones_like(distribution_from_text.scale)
            distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)
        else:
            distribution_ref = None
        # Compute the losses
        loss = self.losses[split].update(ds_text=datastruct_from_text,
                                        ds_motion=datastruct_from_motion,
                                        ds_ref=datastruct_ref,
                                        lat_text=latent_from_text,
                                        lat_motion=latent_from_motion,
                                        dis_text=distribution_from_text,
                                        dis_motion=distribution_from_motion,
                                        dis_ref=distribution_ref)
        if split == "val":
            # Compute the metrics
            self.metrics.update(datastruct_from_text.detach().joints,
                                datastruct_ref.detach().joints,
                                batch["length"])
        # render_list_train = ['1574-1', '7286-0', '6001-0', '4224-2', '3415-0', '2634-0', '2424-1', '4550-0']
        # render_list_val = ['2307-1', '6078-0', '5210-0', '12255-0', '11346-2', '11671-1', '443-8', '3290-3', '2014-0', '973-12']
        # breakpoint()
        # # if batch['keyid']
        # # self.store_examples[] = joints_input
        # keys_val = list(set(batch['keyid']) & set(render_list_val))
        # keys_train = list(set(batch['keyid']) & set(render_list_train))
        # if keys_val:
        #     for kv in keys_val:
        #         .index(kv)
        #         self.store_examples['val'] = {'text': []append(batch)
        # if keys_train:
        
        # TODO fix this bring back the groundtruth somehow 
        # TODO eg. create a dataset only for the keys we care about
        
        # if batch_idx == 0:
        #     nvids = self.hparams.nvids_to_save
        #     if nvids is not None and nvids != 0:
        #         del self.store_examples[split]
        #         lengths = batch["length"][:nvids]
        #         def prepare_pos(x):
        #             x = x.detach().joints[:nvids]
        #             x = x.cpu().numpy()
        #             return remove_padding(x, lengths)
        #         def prepare_verts(x):
        #             x = x.detach().vertices[:nvids]
        #             x = x.cpu().numpy()
        #             return remove_padding(x, lengths)

        #         # ['transforms', '_joints2jfeats', 'features', 'joints_', 'jfeats_']
        #         #
        #         # ['transforms', '_rots2rfeats', '_rots2joints', '_joints2jfeats',
        #         # 'features', 'rots_', 'rfeats_', 'joints_', 'jfeats_']

        #         self.store_examples[split] = { "text": batch["text"][:nvids] }
        #         # if 'vertices_' in datastruct_from_text.keys():
        #         #     # get SMPL features for viz
        #         #     self.store_examples[split].update({
        #         #         "from_text": prepare_verts(datastruct_from_text),
        #         #         # "from_motion": prepare_verts(datastruct_from_motion),
        #         #         "ref": prepare_verts(datastruct_ref),
        #         #     })
        #         # else:
        #         self.store_examples[split].update({
        #             "ref": prepare_pos(datastruct_ref),
        #             "from_text": prepare_pos(datastruct_from_text),
        #             # "from_motion": prepare_pos(datastruct_from_motion)
        #         })
        return loss
