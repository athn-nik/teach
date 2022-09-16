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


class MetaActorDecoder2(pl.LightningModule):
    def __init__(self, nfeats: int,
                 latent_dim: int = 256, ff_size: int = 1024,
                 num_layers: int = 4, num_heads: int = 4,
                 dropout: float = 0.1,
                 mode="posencoding",
                 prev_data_mode="z1t",
                 activation: str = "gelu",
                 hist_frames: int = 1,
                 **kwargs) -> None:

        assert mode in ["memory", "posencoding"]
        assert prev_data_mode in ["z1t", "hist_frame", "hist_frame_outpast"]
        
        super().__init__()
        self.save_hyperparameters(logger=False)

        output_feats = nfeats
        self.hist_frames = hist_frames
        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout,
                                                        negative=True)

        input_feats = nfeats
        self.skel_embedding = nn.Linear(input_feats, latent_dim)

        # For the first motion => learnable token
        self.first_motion_token = nn.Parameter(torch.randn(latent_dim))

        seq_trans_decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim,
                                                             nhead=num_heads,
                                                             dim_feedforward=ff_size,
                                                             dropout=dropout,
                                                             activation=activation)

        self.seqTransDecoder = nn.TransformerDecoder(seq_trans_decoder_layer,
                                                     num_layers=num_layers)

        self.final_layer = nn.Linear(latent_dim, output_feats)

    def forward(self, z: Tensor, prev_data: Tensor, lengths: List[int], debug=False):
        # breakpoint()
        mask = lengths_to_mask(lengths, z.device)
        latent_dim = z.shape[1]
        bs, nframes = mask.shape
        nfeats = self.hparams.nfeats

        # prev_data for pairs of actions
        # (1): z_T of the previous action
        # (2): history features from the previous pose

        if prev_data is None:
            # take a learnable embedding
            prev_data_emb = self.first_motion_token[None][None] 
            if self.hparams.prev_data_mode in ["hist_frame", "hist_frame_outpast"]:
                prev_data_emb = prev_data_emb.repeat((self.hist_frames, bs, 1))

            elif self.hparams.prev_data_mode == "z1t":
                prev_data_emb = prev_data_emb.repeat((1, bs, 1))

            # prev_data_emb: [hist_frame or 1, batch, latent_dim:64]       
        elif self.hparams.prev_data_mode == "z1t":
            prev_data_emb = prev_data[None]    
            # [1, batch, latent_dim]        
        elif self.hparams.prev_data_mode in ["hist_frame", "hist_frame_outpast"]:
            # prev_data: [batch, hist_frame, motion_feats:64] 
            prev_data_emb = self.skel_embedding(prev_data).permute(1, 0, 2)
            # prev_data_emb: [hist_frame, batch, latent_dim:64]

        if self.hparams.mode == "memory":
            # two elements in the memory
            # the latent vector
            # the embedding of the first frame
            # concatenate all the history frames 
            if self.hparams.prev_data_mode == "z1t":
                # memory_time_queries = torch.zeros(2, bs, latent_dim, device=memory.device)                
                memory = torch.stack((*prev_data_emb, z))
                memory = self.sequence_pos_encoding(memory, hist_frames=-1)                
            else: # "hist_frame"
                prev_data_emb = self.sequence_pos_encoding(prev_data_emb, hist_frames=self.hist_frames)
                memory = torch.stack((z, *prev_data_emb))
        else:
            memory = z[None]  # sequence of 1 element for the memory

        # Construct time queries
        time_queries = torch.zeros(nframes, bs, latent_dim, device=memory.device)
        
        if self.hparams.mode == "memory":
            queries = self.sequence_pos_encoding(time_queries)
        elif self.hparams.mode == "posencoding":                        
            if self.hparams.prev_data_mode in ["hist_frame", "hist_frame_outpast"]:
                queries = self.sequence_pos_encoding(torch.cat((prev_data_emb, time_queries), 
                                                               axis=0),
                                                               hist_frames=self.hist_frames)
            else:            
                time_queries = self.sequence_pos_encoding(time_queries)            
                queries = torch.cat((prev_data_emb,
                                     time_queries), axis=0)
                
            # create a bigger mask
            prev_data_mask = torch.ones((bs, len(prev_data_emb)),
                                        dtype=bool, device=z.device)
            mask = torch.cat((prev_data_mask, mask), 1)
        
        # Pass through the transformer decoder
        # with the latent vector for memory
        output = self.seqTransDecoder(tgt=queries, memory=memory,
                                      tgt_key_padding_mask=~mask)

        output = self.final_layer(output)
        # zero for padded area
        output[~mask.T] = 0

        if self.hparams.mode == "posencoding":
            # remove the first one, as it is only to give clues
            if prev_data is None or self.hparams.prev_data_mode != "hist_frame_outpast":
                output = output[len(prev_data_emb):]

        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)
        return feats
