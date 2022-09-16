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

from typing import List, Union
from torch import nn, Tensor

from teach.model.utils import PositionalEncoding
from teach.data.tools import lengths_to_mask


class ActionLevelTextEncoder(DistilbertEncoderBase):
    def __init__(self, modelpath: str,
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

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=num_heads,
                                                             dim_feedforward=ff_size,
                                                             dropout=dropout,
                                                             activation=activation)

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer,
                                                     num_layers=num_layers)

    def forward(self, actions: List[List[str]]) -> Tensor:
        # Space between words
        actions_texts = [[" ".join(action) for action in actions_batch] for actions_batch in actions]

        # Stack all of them together: made only one pass to the text encoder
        # keep the indexes, for later uses
        text_and_idx = [(action, y)
                        for actions, y in zip(actions_texts, range(len(actions_texts)))
                        for action in actions]

        texts, indices = zip(*text_and_idx)
        text_encoded, mask = self.get_last_hidden_state(list(texts), return_mask=True)

        x = self.projection(text_encoded)

        # extra notes: here bs/batch_size is the total number of sentences
        # it is not the same as the "training" batch_size (len(actions))
        bs, nframes, _ = x.shape

        # bs, nframes, totjoints, nfeats = x.shape
        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        if self.hparams.vae:
            mu_token = torch.tile(self.mu_token, (bs,)).reshape(bs, -1)
            logvar_token = torch.tile(self.logvar_token, (bs,)).reshape(bs, -1)

            # adding the embedding token for all sequences
            xseq = torch.cat((mu_token[None], logvar_token[None], x), 0)

            # create a bigger mask, to allow attend to emb
            token_mask = torch.ones((bs, 2), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)
        else:
            emb_token = torch.tile(self.emb_token, (bs,)).reshape(bs, -1)

            # adding the embedding token for all sequences
            xseq = torch.cat((emb_token[None], x), 0)

            # create a bigger mask, to allow attend to emb
            token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)

        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)

        if self.hparams.vae:
            mu, logvar = final[0], final[1]
            std = logvar.exp().pow(0.5)

            # extract each mu/std for each sentences
            actions_mu = [[] for _ in actions]
            actions_std = [[] for _ in actions]
            for i, index in enumerate(indices):
                actions_mu[index].append(mu[i])
                actions_std[index].append(std[i])

            actions_dist = [torch.distributions.Normal(torch.stack(action_mu), torch.stack(action_std))
                            for action_mu, action_std in zip(actions_mu, actions_std)]
            return actions_dist

        else:
            emb = final[0]
            # extract each embedding for each sentences
            actions_emb = [[] for _ in actions]
            for i, index in enumerate(indices):
                actions_emb[index].append(emb[i])

            actions_emb = [torch.stack(action_emb) for action_emb in actions_emb]
            return actions_emb
