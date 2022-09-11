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
        elif return_type == "joints":
            return final_datastruct.joints
        else:
            raise NotImplementedError


    def forward(self, batch: dict, return_rots: bool = False) -> List[Tensor]:
        raise NotImplementedError
        # TODO
        # remove datastruct etc etc
        # concatenate the output features: m0 and m1_trans => M
        # transforms M => joints / or anything
        #
        length_0, length_1 = batch["length_0"], batch["length_1"]

        length_transition = batch["length_transition"]
        length_1_with_transition = batch["length_1_with_transition"]
        # motion1_from_text = self.text_to_motion_forward(text_0, length_0, return_latent=True)

        ### TEXT branches        
        text_0, text_1 = batch["text_0"], batch["text_1"]
        # breakpoint()
        #### Encode the texts
        latent_vector_0_T, _ = self.encode_data(text_0, return_latent=True)

        #### Decode into motions
        ## First motion pass
        # Decode the latent vector to a motion
        output_features_0_T = self.motiondecoder(latent_vector_0_T, lengths=length_0)
        output_features_0_T_lst = [feats[:len0] for feats, len0 in zip(output_features_0_T, length_0)]

        # hist_frames
        if self.hist_frames > 0:
            hist_frames_tensor = torch.stack([x[-self.hist_frames:] for x in output_features_0_T_lst])
        else:
            hist_frames_tensor = None

        if self.previous_latent:
            latent_vector_1_T, distribution_1_T = self.encode_data(text_1, hframes=hist_frames_tensor, z_previous=latent_vector_0_T, 
                                                                   return_latent=True)
        else:
            latent_vector_1_T, distribution_1_T = self.encode_data(text_1, hframes=hist_frames_tensor, return_latent=True)

        ## Second motion pass
        # Decode the second latent vector to a motion
        # by taking clues from the last motion            
        output_features_1_T_with_transition = self.motiondecoder(latent_vector_1_T, lengths=length_1_with_transition)
        
        output_features_1_T_with_transition_lst = [feats[:(len1+trans_len)] for feats, trans_len, len1 in zip(output_features_1_T_with_transition, length_transition, length_1)]
        full_mot_batch_T = [torch.cat((mot1, mot2)) for mot1, mot2 in zip(output_features_0_T_lst, output_features_1_T_with_transition_lst)]                    
            
        ### Motion branches     
        if self.use_motion_encoder:   
            motion_0, motion_1 = batch["motion_0"], batch["motion_1"]
            
            #### Encode the motions
            latent_vector_0_M, _ = self.encode_data(motion_0, return_latent=True)

            #### Decode into motions
            ## First motion pass
            # Decode the latent vector to a motion
            output_features_0_M = self.motiondecoder(latent_vector_0_M, lengths=length_0)
            output_features_0_M_lst = [feats[:len0] for feats, len0 in zip(output_features_0_M, length_0)]

            # TODO: maybe also use the previous motion 
            latent_vector_1_M, distribution_1_M = self.encode_data(motion_1, return_latent=True)
            
            output_features_1_M_with_transition = self.motiondecoder(latent_vector_1_M, lengths=length_1_with_transition)
            
            output_features_1_M_with_transition_lst = [feats[:(len1+trans_len)] for feats, trans_len, len1 in zip(output_features_1_M_with_transition, length_transition, length_1)]
            full_mot_batch_M = [torch.cat((mot1, mot2)) for mot1, mot2 in zip(output_features_0_M_lst, output_features_1_M_with_transition_lst)]                    
            
                                
        if return_rots:
            full_mot = [self.Datastruct(features=motion).rots for motion in full_mot_batch]
            return full_mot

        full_mot = [self.Datastruct(features=motion).joints for motion in full_mot_batch]

        return full_mot


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

        # Encode the text/decode to a motion
        # length
        # datastruct
        # text => concatenatation of texts (commas)
        #

        # sequences of lengths (M1, T, M2)
        # is_transition_included = True/False

        # datastruct
        # text => [t1, t2]

        # first_frames = batch["datastruct"].features[:, 0, :]

        # [M1, M2]
        # => M2 | M1 => optimize
        #  -> one case  prev_data = z1_t
        #               prev_data = H_F1 / H_hat
        # => M1  => optimize
        #
        # act1 / act2
        #
        # len(batch["text"]) == 2
        # batch["text"][0] => first texts (in each batch)
        #
        # batch["text"]
        # -> [act1, act2]
        #
        #
        #
        # len(batch["lengths"]) == 2
        # batch["lengths"][1] should be either lengths of M2 or T+M2
        # batch["transition_lenght"] = 0 or the value
        #
        #

        # self.rots2rfeats = rots2rfeats
        # self.rots2joints = rots2joints
        # self.joints2jfeats = joints2jfeats

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

        # loss M1 / M2, eval M1 / M2      -> current
        # loss M1+trans+M2, eval M1 / M2
        # loss M1 / M2, eval M1+trans+M2
        # loss M1+trans+M2, eval M1+trans+M2

        #     # Compute the metrics
        #     # breakpoint()
        #     # output_features_1_T_lst[0]  -> F, feats -> F, xyz
        # if split == "val":# or batch_idx == 0:
        #     self.transforms.rots2rfeats.to(latent_vector_0_T.device)
        #     self.transforms.rots2joints.to(latent_vector_0_T.device)

        #     if not self.hparams.losses.loss_on_transition:
        #         joints_0 = [self.transforms.rots2joints(self.transforms.rots2rfeats.inverse(x.detach())) for x in output_features_0_T_lst]
        #         joints_1 = [self.transforms.rots2joints(self.transforms.rots2rfeats.inverse(x.detach())) for x in output_features_1_T_lst]
        #         ref_joints_0 = [self.transforms.rots2joints(self.transforms.rots2rfeats.inverse(x.detach())) for x in input_motion_feats_0_lst]
        #         ref_joints_1 = [self.transforms.rots2joints(self.transforms.rots2rfeats.inverse(x.detach())) for x in input_motion_feats_1_lst]

        #         self.metrics_0.update(joints_0.detach(), ref_joints_0.detach(), length_0.detach())
        #         self.metrics_1.update(joints_1.detach(), ref_joints_1.detach(), length_1.detach())

        #     else:
        #         # if self.motiondecoder.hparams.prev_data_mode == "hist_frame_outpast":
        #         #     # remove the hframes
        #         #     # M0 / M1[hframes:]
        #         #     breakpoint()
        #         #     joints_output = [self.transforms.rots2joints(self.transforms.rots2rfeats.inverse(torch.cat((x, y[hframes:]))))
        #         #                      for x, y in zip(output_features_0_T_lst, output_features_1_T_with_transition_lst)]
        #         # else:
        #         # Evalute on all the joints (concatenation)
        #         joints_output = [self.transforms.rots2joints(self.transforms.rots2rfeats.inverse(torch.cat((x.detach(), y.detach()))))
        #                             for x, y in zip(output_features_0_T_lst, output_features_1_T_with_transition_lst)]

        #         joints_input = [self.transforms.rots2joints(self.transforms.rots2rfeats.inverse(torch.cat((x.detach(), y.detach()))))
        #                         for x, y in zip(input_motion_feats_0_lst, input_motion_feats_1_with_transition_lst)]

        #         self.metrics.update(joints_output, joints_input, total_length)

        # render_list_train = ['1574-1', '7286-0', '6001-0', '4224-2', '3415-0', '2634-0', '2424-1', '4550-0']
        # render_list_val = ['2307-1', '6078-0', '5210-0', '12255-0', '11346-2', '11671-1', '443-8', '3290-3', '2014-0', '973-12']
        # breakpoint()
        # # if batch['keyid']
        # # self.store_examples[] = joints_input
        # keys_val = list(set(batch['keyid']) & set(render_list_val))
        # keys_train = list(set(batch['keyid']) & set(render_list_train))

        # if batch_idx == 0:
        #     nvids = self.hparams.nvids_to_save
        #     if nvids is not None and nvids != 0:
        #         del self.store_examples[split]
        #         # lengths = batch["length"][:nvids]
        #         # length_0, length_1 = batch["length_0"], batch["length_1"]
        #         # length_transition = batch["length_transition"]
        #         # length_1_with_transition = batch["length_1_with_transition"]

        #         def prepare_pos(x, lens):
        #             x = x.detach().joints[:nvids]
        #             x = x.cpu().numpy()
        #             return remove_padding(x, lens)
        #         def prepare_rots(x, lens):
        #             x = x.detach().rots.rots[:nvids]
        #             x = x.cpu().numpy()
        #             return remove_padding(x, lens)
        #         def prepare_trans(x, lens):
        #             x = x.detach().rots.trans[:nvids]
        #             x = x.cpu().numpy()
        #             return remove_padding(x, lens)
        #         texts = list(zip(batch['text_0'], batch['text_1']))
        #         texts = [', '.join(t_tup) for t_tup in texts]

        #         self.store_examples[split] = {
        #             "text": texts[:nvids],
        #             "ref": joints_input[:nvids],
        #             "from_text": joints_output[:nvids],
        #             # "from_motion": prepare_pos(datastruct_from_motion),

        #             # get SMPL features for viz
        #             # "from_text_rots": prepare_rots(datastruct_from_text),
        #             # "from_text_trans": prepare_trans(datastruct_from_text),
        #             # "from_motion_rots": prepare_rots(datastruct_from_motion),
        #             # "from_motion_trans": prepare_trans(datastruct_from_motion),

        #             # get SMPL groundtruth for viz
        #             # "ref_rots": prepare_rots(datastruct_ref),
        #             # "ref_trans": prepare_trans(datastruct_ref)
        #         }

        # if batch_idx == 0:
        #     nvids = self.hparams.nvids_to_save
        #     if nvids is not None and nvids != 0:
        #         del self.store_examples[split]
        #         lengths = batch["length"][:nvids]

        #         def prepare(x):
        #             x = x.detach().joints[:nvids]
        #             x = x.cpu().numpy()
        #             return remove_padding(x, lengths)

        #         self.store_examples[split] = {
        #             "text": batch["text"][:nvids],
        #             "ref": prepare(pose_data_ref),
        #             "from_text": prepare(pose_data_from_text),
        #             "from_motion": prepare(pose_data_from_motion)
        #         }

        return loss
