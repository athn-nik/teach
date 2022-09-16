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

from .distilbert import DistilbertEncoderBase
import torch

from typing import List, Union, Optional
from torch import nn, Tensor
from torch.distributions.distribution import Distribution

from teach.model.utils import PositionalEncoding
from teach.data.tools import lengths_to_mask


class TextHist(DistilbertEncoderBase):
    def __init__(self, modelpath: str,
                 nfeats: int,
                 hist_frames: int = 1,
                 finetune: bool = False,
                 vae: bool = True,
                 latent_dim: int = 256,
                 ff_size: int = 1024,
                 num_layers: int = 4, num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu", **kwargs) -> None:
        super().__init__(modelpath=modelpath, finetune=finetune)
        self.save_hyperparameters(logger=False)

        encoded_dim = self.text_encoded_dim

        # Projection of the text-outputs into the latent space
        self.projection = nn.Sequential(nn.ReLU(),
                                        nn.Linear(encoded_dim, latent_dim))

        # TransformerVAE adapted from ACTOR
        # Action agnostic: only one set of params
        if vae:
            self.mu_token = nn.Parameter(torch.randn(latent_dim))
            self.logvar_token = nn.Parameter(torch.randn(latent_dim))
        else:
            self.emb_token = nn.Parameter(torch.randn(latent_dim))

        self.seperation_token = nn.Parameter(torch.randn(latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=num_heads,
                                                             dim_feedforward=ff_size,
                                                             dropout=dropout,
                                                             activation=activation)

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer,
                                                     num_layers=num_layers)

        motion_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=ff_size,
                                                          dropout=dropout,
                                                          activation=activation)
        input_feats = nfeats
        self.skel_embedding = nn.Linear(input_feats, latent_dim)
        self.motionTransEncoder = nn.TransformerEncoder(motion_encoder_layer, num_layers=num_layers)
        # TODO
        self.hist_frames = hist_frames

    def forward(self, texts: List[str], hframes: Optional[Tensor] = None, z_pt: Optional[Tensor] = None) -> Union[Tensor, Distribution]:
        # TEXT part
        text_encoded, mask = self.get_last_hidden_state(texts, return_mask=True)
        x = self.projection(text_encoded)
        bs, nframes, _ = x.shape
        # bs, nframes, totjoints, nfeats = x.shape
        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nwords, bs, latent_dim]

        # add positional encoding
        x = self.sequence_pos_encoding(x)

        # MOTION part
        if hframes is not None:
            # hist_frames : (batch, X frames, features)
            hframes_emb = self.skel_embedding(hframes)
            hframes_emb = hframes_emb.permute(1, 0, 2)
            # hframes_emb : (X frames, batch, features)
            hframes_emb = self.sequence_pos_encoding(hframes_emb)
            hframes_emb = self.seqTransEncoder(hframes_emb)

        # TEXT-MOTION part
        # VAE only for now
        if self.hparams.vae:
            mu_token = torch.tile(self.mu_token, (bs,)).reshape(bs, -1)
            logvar_token = torch.tile(self.logvar_token, (bs,)).reshape(bs, -1)
            sep_token = torch.tile(self.seperation_token, (bs,)).reshape(bs, -1)
            if hframes is not None and z_pt is not None:
                # adding the distribution tokens for all sequences
                xseq = torch.cat((mu_token[None], logvar_token[None], hframes_emb, z_pt[None], sep_token[None], x), 0)
                number_of_extra_tokens = 2 + self.hist_frames + 1 + 1

            elif hframes is not None:
                # adding the distribution tokens for all sequences
                xseq = torch.cat((mu_token[None], logvar_token[None], hframes_emb, sep_token[None], x), 0)
                number_of_extra_tokens = 2 + self.hist_frames + 1
            elif z_pt is not None:
                xseq = torch.cat((mu_token[None], logvar_token[None], z_pt[None], sep_token[None], x), 0)
                number_of_extra_tokens = 2 + 1 + 1
            else:
                xseq = torch.cat((mu_token[None], logvar_token[None], sep_token[None], x), 0)
                # create a bigger mask, to allow attend to mu and logvar
                number_of_extra_tokens = 3
            token_mask = torch.ones((bs, number_of_extra_tokens), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)
        else:
            emb_token = torch.tile(self.emb_token, (bs,)).reshape(bs, -1)
            sep_token = torch.tile(self.seperation_token, (bs,)).reshape(bs, -1)

            if hframes is not None:
                # adding the embedding token for all sequences
                xseq = torch.cat((emb_token[None], hframes_emb, sep_token[None], x), 0)
                number_of_extra_tokens = 1 + self.hist_frames + 1
            else:
                xseq = torch.cat((emb_token[None], sep_token[None], x), 0)
                number_of_extra_tokens = 2
            # create a bigger mask, to allow attend to emb
            token_mask = torch.ones((bs, number_of_extra_tokens), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)

        if self.hparams.vae:
            mu, logvar = final[0], final[1]
            std = logvar.exp().pow(0.5)
            # https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
            try:
                dist = torch.distributions.normal.Normal(mu, std)
            except ValueError:
                import ipdb; ipdb.set_trace()  # noqa
                pass
            return dist
        else:
            return final[0]
