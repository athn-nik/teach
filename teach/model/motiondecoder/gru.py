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
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from typing import List, Optional
from torch import nn, Tensor

from teach.model.utils import PositionalEncoding
from teach.data.tools import lengths_to_mask


class GRUDecoder(pl.LightningModule):
    def __init__(self, nfeats: int,
                 latent_dim: int = 256,
                 num_layers: int = 4, **kwargs) -> None:

        super().__init__()
        self.save_hyperparameters(logger=False)

        output_feats = nfeats
        # self.feats_embedding = nn.Linear(self.input_feats, latent_dim)

        self.emb_layer = nn.Linear(latent_dim+1, latent_dim)
        self.gru = nn.GRU(latent_dim, latent_dim, num_layers=num_layers)
        self.final_layer = nn.Linear(latent_dim, output_feats)

    def forward(self, z: Tensor, lengths: List[int]):
        mask = lengths_to_mask(lengths, z.device)
        latent_dim = z.shape[1]
        bs, nframes = mask.shape
        nfeats = self.hparams.nfeats

        lengths = torch.tensor(lengths, device=z.device)

        # Repeat the input
        z = z[None].repeat((nframes, 1, 1))

        # Add time information to the input
        time = mask * 1/(lengths[..., None]-1)
        time = (time[:, None] * torch.arange(time.shape[1], device=z.device))[:, 0]
        time = time.T[..., None]
        z = torch.cat((z, time), 2)

        # emb to latent space again
        z = self.emb_layer(z)

        # pass to gru
        z = self.gru(z)[0]
        output = self.final_layer(z)

        # zero for padded area
        output[~mask.T] = 0
        # Pytorch GRU: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)

        return feats
