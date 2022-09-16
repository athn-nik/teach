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


class MetaActorDecoder(pl.LightningModule):
    def __init__(self, nfeats: int,
                 latent_dim: int = 256, ff_size: int = 1024,
                 num_layers: int = 4, num_heads: int = 4,
                 dropout: float = 0.1,
                 mode="posencoding",
                 activation: str = "gelu", **kwargs) -> None:

        assert mode in ["memory", "posencoding"]
        super().__init__()
        self.save_hyperparameters(logger=False)

        output_feats = nfeats

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

        input_feats = nfeats
        self.skel_embedding = nn.Linear(input_feats, latent_dim)

        seq_trans_decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim,
                                                             nhead=num_heads,
                                                             dim_feedforward=ff_size,
                                                             dropout=dropout,
                                                             activation=activation)

        self.seqTransDecoder = nn.TransformerDecoder(seq_trans_decoder_layer,
                                                     num_layers=num_layers)

        self.final_layer = nn.Linear(latent_dim, output_feats)

    def forward(self, z: Tensor, first_frames: List[Tensor], lengths: List[int]):
        mask = lengths_to_mask(lengths, z.device)
        latent_dim = z.shape[1]
        bs, nframes = mask.shape
        nfeats = self.hparams.nfeats

        fframes_emb = self.skel_embedding(first_frames)

        if self.hparams.mode == "memory":
            # two elements in the memory
            # the latent vector
            # the embedding of the first frame
            memory = torch.stack((z, fframes_emb))
        else:
            memory = z[None]  # sequence of 1 element for the memory

        # Construct time queries
        time_queries = torch.zeros(nframes, bs, latent_dim, device=memory.device)
        time_queries = self.sequence_pos_encoding(time_queries)

        if self.hparams.mode == "posencoding":
            queries = torch.cat((fframes_emb[None], time_queries), axis=0)
            # create a bigger mask
            fframes_mask = torch.ones((bs, 1), dtype=bool, device=z.device)
            mask = torch.cat((fframes_mask, mask), 1)
        else:
            queries = time_queries

        # Pass through the transformer decoder
        # with the latent vector for memory
        output = self.seqTransDecoder(tgt=queries, memory=memory,
                                      tgt_key_padding_mask=~mask)

        output = self.final_layer(output)
        # zero for padded area
        output[~mask.T] = 0

        if self.hparams.mode == "posencoding":
            # remove the first one, as it is only to give clues
            output = output[1:]

        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)
        return feats
