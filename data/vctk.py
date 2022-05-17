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


class VCTKDataset(BaseDictDataset):
    def __init__(self, *args, **kwargs):
        super(VCTKDataset, self).__init__(*args, **kwargs)

    def getitem(self, item, file=None, group_name=None):
        fragment = self.get_fragment(file)
        return fragment
    

class RandomLoader(Dataset):
    def __init__(self, n_samples, sample_size):
        self.n_samples = n_samples
        self.sample_size = sample_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        item = item % self.n_samples

        clip1 = torch.zeros(self.sample_size)
        #clip2 = torch.zeros(100000)
        #clip_more_neg = torch.tensor(0)
        #group_name = 'no'

        return clip1#, clip2, clip_more_neg, group_name




class VCTKDataModule(BaseDataModule):

    def __init__(self, 
                use_random_loader = False,
                **kwargs):

        super(VCTKDataModule, self).__init__(**kwargs)
        
        self.use_random_loader = use_random_loader
        self.custom_transforms =Compose([
                                RandomApply(
                                    lambda x: random_phase_mangle(x, 20, 2000, .99, self.sr),
                                    p=.8,
                                ),
                                Dequantize(16),
                                lambda x: (Gain(min_gain_in_db=-6, max_gain_in_db=0, 
                                                p=0.5)(x, sample_rate = self.sr)),
                                lambda x: x.astype(np.float32),
                            ])

    def train_dataloader(self):
            print('train loader')
            if self.use_random_loader:
                return DataLoader(RandomLoader(n_samples=30000, sample_size=self.nr_samples))
            else:
                return DataLoader(VCTKDataset(self.groups_train, 
                                                nr_samples=self.nr_samples, # For 4s, nr_samples is sr*4
                                                normalize=self.normalize,
                                                augmentations={"custom_transform": self.custom_transforms }, 
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
        if self.use_random_loader:
            return DataLoader(RandomLoader(n_samples=30000, sample_size=self.nr_samples))
        else:
            return DataLoader(VCTKDataset(self.groups_eval,
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
