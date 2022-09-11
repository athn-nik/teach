import torch
from torch.nn.functional import smooth_l1_loss


class Recons:
    def __call__(self, input_motion_feats_lst, output_features_lst):
        recons = torch.stack([smooth_l1_loss(x, y, reduce="mean") for x,y in zip(input_motion_feats_lst, output_features_lst)]).mean()
        return recons 

    def __repr__(self):
        return "Recons()"
