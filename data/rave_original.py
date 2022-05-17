import torch
from torch.utils.data import DataLoader, random_split, Dataset

from rave.core import random_phase_mangle, EMAModelCheckPoint

import numpy as np

from udls import SimpleDataset, simple_audio_preprocess

from udls.transforms import Compose, RandomApply, Dequantize, RandomCrop

import numpy as np
import os
from tqdm import tqdm
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class RAVEDataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_dir: str = None, 
                 batch_size: int = 32,
                 batch_size_val: int = 32,
                 num_workers: int = 8,
                 sr: int = 44100,
                 n_signal: int = 65536,
                 preprocessed: str = None):

        super().__init__()
        self.wav = data_dir
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.num_workers = num_workers
        self.sr = sr
        self.n_signal = n_signal
        self.preprocessed = preprocessed

    def setup(self, stage = None):

        self.dataset = SimpleDataset(
            self.preprocessed,
            self.wav,
            preprocess_function=simple_audio_preprocess(self.sr,
                                                        2 * self.n_signal),
            split_set="full",
            transforms=Compose([
                RandomCrop(self.n_signal),
                RandomApply(
                    lambda x: random_phase_mangle(x, 20, 2000, .99, self.sr),
                    p=.8,
                ),
                Dequantize(16),
                lambda x: x.astype(np.float32),
            ]),
            )

        val = max((2 * len(self.dataset)) // 100, 1)
        train = len(self.dataset) - val
        self.train, self.val = random_split(
            self.dataset,
            [train, val],
            #generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, True, drop_last=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, self.batch_size_val, False, num_workers=self.num_workers)


    