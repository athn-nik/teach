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

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from teach.data.tools.collate import collate_pairs_and_text, collate_datastruct_and_text


class BASEDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int,
                 num_workers: int,
                 datatype: None):
        super().__init__()
        
        if datatype == 'separate_pairs':
            collate_fn = collate_pairs_and_text
        else:
            collate_fn = collate_datastruct_and_text
            
        self.dataloader_options = {"batch_size": batch_size, "num_workers": num_workers,
                                   "collate_fn": collate_fn}
        # need to be overloaded:
        # - self.Dataset
        # - self._sample_set => load only a small subset
        #   There is an helper below (get_sample_set)
        # - self.nfeats
        # - self.transforms

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

        # Optional
        self._subset_dataset = None

    def get_sample_set(self, overrides={}):
        sample_params = self.hparams.copy()
        sample_params.update(overrides)
        return self.Dataset(**sample_params)

    @property
    def train_dataset(self):
        if self._train_dataset is None:
            self._train_dataset = self.Dataset(split="train", **self.hparams)
        return self._train_dataset

    @property
    def val_dataset(self):
        if self._val_dataset is None:
            self._val_dataset = self.Dataset(split="val", **self.hparams)
        return self._val_dataset

    @property
    def test_dataset(self):
        if self._test_dataset is None:
            self._test_dataset = self.Dataset(split="test", **self.hparams)
        return self._test_dataset

    @property
    def subset_dataset(self):
        if self._subset_dataset is None:
            self._subset_dataset = self.Dataset(split="subset", **self.hparams)
        return self._subset_dataset

    def setup(self, stage=None):
        # Use the getter the first time to load the data
        if stage in (None, "fit"):
            _ = self.train_dataset
            _ = self.val_dataset
        if stage in (None, "test"):
            _ = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.dataloader_options)

    def predict_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=False, **self.dataloader_options)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, **self.dataloader_options)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, **self.dataloader_options)

    def subset_dataloader(self):
        return DataLoader(self.subset_dataset, shuffle=False, **self.dataloader_options)
