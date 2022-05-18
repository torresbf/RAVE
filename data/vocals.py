import torch
from torch.utils.data import DataLoader, random_split, Dataset

from rave.core import random_phase_mangle, EMAModelCheckPoint

import numpy as np
from udls.transforms import Compose, RandomApply, Dequantize, RandomCrop
from audiomentations import Gain

import numpy as np
import os
from tqdm import tqdm
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import json

from data.base import BaseDictDataset, BaseDataModule
from utils.augmentations import aug


class VocalsDataset(BaseDictDataset):
    def __init__(self, *args, **kwargs):
        super(VocalsDataset, self).__init__(*args, **kwargs)

    def getitem(self, item, file=None, group_name=None):
        fragment = self.get_fragment(file)
        return fragment


class VocalsDataModule(BaseDataModule):

    def __init__(self,
                **kwargs):

        super(VocalsDataModule, self).__init__(**kwargs)
        
        
    def train_dataloader(self):
        print('train loader')
        return DataLoader(VocalsDataset(self.groups_train, 
                                            nr_samples=self.nr_samples, # For 4s, nr_samples is sr*4
                                            normalize=self.normalize,
                                            augmentations=self.augs, 
                                            transform_override=self.transform_override,
                                            batch_sampling_mode=self.batch_sampling_mode,
                                            sr = self.sr,
                                            multi_epoch=1),
                                shuffle=True, 
                                batch_size=self.batch_size,
                                num_workers=self.num_workers, 
                                drop_last=True)

    def val_dataloader(self):
        print('val loader')
        return DataLoader(VocalsDataset(self.groups_eval,
                                        nr_samples=self.nr_samples, # For 4s, nr_samples is sr*4
                                        normalize=self.normalize,
                                        augmentations={},
                                        # positive_examples=self.positive_examples,
                                        batch_sampling_mode=self.batch_sampling_mode,
                                        sr = self.sr,
                                        multi_epoch=1),
                            shuffle=False, 
                            batch_size=self.batch_size_val,
                            num_workers=self.num_workers, 
                            drop_last=False)

    def prepare_data_end(self):
        print(f"Augmentations: {json.dumps(self.augs, sort_keys=True, indent=4)}")
