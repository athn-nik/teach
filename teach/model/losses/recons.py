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

import torch
from torch.nn.functional import smooth_l1_loss


class Recons:
    def __call__(self, input_motion_feats_lst, output_features_lst):
        recons = torch.stack([smooth_l1_loss(x, y, reduce="mean") for x,y in zip(input_motion_feats_lst, output_features_lst)]).mean()
        return recons 

    def __repr__(self):
        return "Recons()"
