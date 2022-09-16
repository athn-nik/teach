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

from typing import List, Union
import pytorch_lightning as pl

import torch.nn as nn
import os

import torch
from torch import Tensor
from torch.distributions.distribution import Distribution

from .distilbert import DistilbertEncoderBase


class DistilbertEncoderLinear(DistilbertEncoderBase):
    def __init__(self, modelpath: str,
                 finetune: bool = False,
                 vae: bool = True,
                 latent_dim: int = 256,
                 **kwargs) -> None:
        super().__init__(modelpath=modelpath, finetune=finetune)
        self.save_hyperparameters(logger=False)

        encoded_dim = self.text_encoded_dim
        # Train a embedding layer
        if vae:
            self.mu_projection = nn.Sequential(nn.ReLU(),
                                               nn.Linear(encoded_dim, latent_dim))
            self.logvar_projection = nn.Sequential(nn.ReLU(),
                                                   nn.Linear(encoded_dim, latent_dim))
        else:
            self.projection = nn.Sequential(nn.ReLU(),
                                            nn.Linear(encoded_dim, latent_dim))

    def forward(self, texts: List[str]) -> Union[Tensor, Distribution]:
        last_hidden_state = self.get_last_hidden_state(texts)
        latent = last_hidden_state[:, 0]

        if self.hparams.vae:
            mu = self.mu_projection(latent)
            logvar = self.logvar_projection(latent)
            std = logvar.exp().pow(0.5)
            # https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
            return torch.distributions.Normal(mu, std)
        else:
            return self.projection(latent)
