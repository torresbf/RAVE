import torch
from torch.utils.data import DataLoader, random_split, Dataset

from utils.augmentations import aug
from utils.data_utils import prepare_fn_groups_vocal, filter1_voice_wav, get_fragment_from_file

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
import json

from utils.augmentations import aug
from utils.data_utils import get_fragment_from_file, prepare_fn_groups_vocal, filter1_voice_wav



class DictLoader(Dataset):
    def __init__(self, 
                groups:dict, 
                nr_samples: int, 
                normalize: bool = True, 
                augmentations_pos:dict = {}, 
                augmentations_neg:dict = {}, 
                transform_override: bool = False,
                positive_examples: str = 'same_clip',
                batch_sampling_mode: str = 'sample_clips',
                multi_epoch: int = 1):
        self.groups = groups
        self.multi_epoch = multi_epoch
        self.nr_samples = nr_samples
        self.normalize = normalize
        self.transform_override = transform_override

        self.augmentations_pos = False
        if augmentations_pos.get('enable', 0):
            if self.transform_override:
                raise ValueError("Transform override but augmentations passed are not the transforms")
            if augmentations_pos['enable']:
                self.augmentations_pos = augmentations_pos
        if self.transform_override:
            self.augmentations_pos = augmentations_pos

        self.augmentations_neg = False
        if augmentations_neg.get('enable', 0):
            if augmentations_neg['enable']:
                self.augmentations_neg = augmentations_neg
            
        self.positive_examples=positive_examples
        self.batch_sampling_mode=batch_sampling_mode

        self.groups_keys = list(self.groups.keys())
        self.inv_map = {fn: k for k, v in groups.items() for fn in v}
        self.inv_map_keys = list(self.inv_map.keys())
        self.inv_map_values = list(self.inv_map.values())

        if self.batch_sampling_mode == 'sample_clips':
            self.data_len = len(self.inv_map)
        else:
            self.data_len = len(self.groups)

    def __len__(self):
        return self.data_len * self.multi_epoch

    def __getitem__(self, item):
        item = item % self.data_len

        if self.batch_sampling_mode == 'sample_clips':
            selec_fn = self.inv_map_keys[item]  # Sample a clip in the dataset
            group_name = self.inv_map_values[item]  # Gets clip artist
        elif self.batch_sampling_mode == 'sample_groups':
            group_name = self.groups_keys[item]  # Sample an artist (group)
        else:
            raise ValueError(f'Invalid batch sampling mode. Value was {self.batch_sampling_mode}')
        
        # Gets list of fns of the artist
        selec_group = self.groups[group_name]  

        file1 = selec_fn if self.batch_sampling_mode == 'sample_clips' else selec_group[np.random.randint(len(selec_group))]
        
        if self.positive_examples == 'same_clip' or group_name == 'unknown':
            file2 = file1
        elif self.positive_examples == 'same_group':
            rand1 = np.random.randint(len(selec_group))
            file2 = selec_group[rand1]

        clip1 = get_fragment_from_file(file1, self.nr_samples, self.normalize, draw_random=True)
        #clip2 = get_fragment_from_file(file2, self.nr_samples, self.normalize, draw_random=True)

        # If one wants to apply custom transforms, pass them through augmentations_pos dict and use transform_override
        override = False
        if self.transform_override:
            override = self.augmentations_pos
        clip1 = aug(np.cast['float32'](clip1), self.augmentations_pos, override=override)
        #lip2 = aug(np.cast['float32'](clip2), self.augmentations_pos)
        # clip_more_neg = aug(np.cast['float32'](clip1), self.augmentations_neg) if self.augmentations_neg else torch.tensor(0)
        return clip1 #, clip2, clip_more_neg, group_name

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


class DataModule(pl.LightningDataModule):
    def __init__(self, 
                 dataset_dirs: list = [],
                 batch_size: int = 32,
                 batch_size_val: int = 32,
                 nr_samples: int = 176000,
                 normalize: bool = True,
                 num_workers: int = 8,
                 positive_examples: str = "same_clip",
                 batch_sampling_mode: str = "sample_clips",
                 eval_frac: float = 0.1,
                 group_name_is_folder: bool = False,
                 group_by_artist : bool = False,
                 mask_samples: bool = False,
                 augs_pos: dict = {},
                 augs_neg: dict = {},
                 transform_override: bool = False,
                 verbose: bool = True,
                 use_random_loader: bool = False
                 ):

        """
        Args:
            dataset_dirs: List of directories where data is contained. Dirs will be scanned and 
                the audio files will be added to the data dictionary
            batch_size: Train loader batch size
            batch_size_val: Validation loader batch size
            nr_samples: Number of samples of fragment to be extracted on data loading. 
                The audio fragment is extracted randomly from a sampled audio file (clip)
            normalize: Wether to normalize or not the extracted fragments
            num_workers: Number of workers
            positive_examples: If "same_clip", a second fragment (positive example) will be drawn from the same clip already sampled 
                               If "same_group", the second fragment will be drawn from another (randomly sampled) clip belonging to the same group
            batch_sampling_mode: If "sample_clips", clips are sampled randomly out of a list containig all clips on the data
                               If "sample_groups", first a group is sampled out of the list of groups, then a clip is sampled from that group
            group_by_artist: If true, assumes each subfolder on dataset directory is a group. 
                All files on this folder will be mapped to the same entry on the groups dictionary (either folder name or 'unknown', 
                depending on group_name_is_folder).
                If false, each file will be mapped to it's own group
            group_name_is_folder: If true, the dict key of a group will be the folder name
            eval_frac: Fraction of data groups to be separated as validation set
            augs_pos: Dict containing augmentations to perform on positive examples and it's probabilities
            augs_neg: Dict containing augmentations to perform on negative examples and it's probabilities
            transform_override: If true, augs_pos will contain a dict of transforms that will replace the positive augmentations loader
            verbose: 
            use_random_loader: Loads a random loader (for debug purposes)
            
        """
        super().__init__()
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.dataset_dirs = dataset_dirs
        self.normalize = normalize
        self.positive_examples = positive_examples
        self.batch_sampling_mode = batch_sampling_mode
        self.group_name_is_folder = group_name_is_folder
        self.group_by_artist = group_by_artist
        self.eval_frac = eval_frac
        self.groups = {}
        self.group_names = []  # maybe to register buffer
        self.augs_pos = augs_pos
        self.augs_neg = augs_neg
        self.transform_override = transform_override
        self.nr_samples = nr_samples
        self.num_workers = num_workers
        self.verbose = verbose
        self.use_random_loader = use_random_loader
    
    def prepare_data(self):
        print('prep data')
        if not self.use_random_loader:
            assert len(self.dataset_dirs) > 0
            # For every dataset (folder with group names in subfolders) do parse
            for dataset in self.dataset_dirs:
                self.groups = prepare_fn_groups_vocal(
                    dataset,
                    groups=self.groups,
                    select_only_groups=['out_44100'],
                    filter_fun_level1=filter1_voice_wav,
                    group_name_is_folder=self.group_name_is_folder,
                    group_by_artist=self.group_by_artist)

            # Get group names
            self.group_names = list(self.groups.keys())
            np.random.shuffle(self.group_names)

            self.n_files = sum([len(group) for group in list(self.groups.values())])
            self.n_groups = len(self.group_names)
            
            if self.verbose:
                print(f'Number of files in dataset: {self.n_files}, split into {self.n_groups} artists')

    def setup(self, stage = None):
        print('setup')

        if not self.use_random_loader:
            count_elements_in_dict_split = lambda dic, keys_subset : sum([len(dic[key]) for key in keys_subset])

            # Train/val split splits groups
            self.eval_split = int(len(self.group_names) * self.eval_frac)
            self.groups_train = dict((k, self.groups[k]) for k in self.group_names[self.eval_split:])
            self.groups_eval = dict((k, self.groups[k]) for k in self.group_names[:self.eval_split])
            self.n_files_train = count_elements_in_dict_split(self.groups, self.group_names[self.eval_split:])
            self.n_files_eval = count_elements_in_dict_split(self.groups, self.group_names[:self.eval_split])

            if self.batch_sampling_mode == 'sample_clips':
                n_batches = self.n_files_train / self.batch_size
            elif self.batch_sampling_mode == 'sample_groups':
                n_batches = self.eval_split / self.batch_size
            else:
                raise ValueError(f'Invalid batch sampling mode. Value was {self.batch_sampling_mode}')

            if self.verbose:
                print(f"Number of training batches: {n_batches}")
                print(f"Size train (groups) : {self.n_groups-self.eval_split}, eval: {self.eval_split}")
                print(f"Size train (files): {self.n_files_train}")
                print(f"Size eval (files): {self.n_files_eval}")

            self.custom_transforms = transforms=Compose([
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
            return DataLoader(DictLoader(self.groups_train, 
                                            nr_samples=self.nr_samples, # For 4s, nr_samples is sr*4
                                            normalize=self.normalize,
                                            augmentations_pos={"custom_transform": self.custom_transforms }, 
                                            augmentations_neg=self.augs_neg,
                                            transform_override=self.transform_override,
                                            positive_examples=self.positive_examples,
                                            batch_sampling_mode=self.batch_sampling_mode,
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
            return DataLoader(DictLoader(self.groups_eval,
                                            nr_samples=self.nr_samples, # For 4s, nr_samples is sr*4
                                            normalize=self.normalize,
                                            augmentations_pos={}, 
                                            augmentations_neg={},
                                            positive_examples=self.positive_examples,
                                            batch_sampling_mode=self.batch_sampling_mode,
                                            multi_epoch=1),
                                shuffle=False, 
                                batch_size=self.batch_size_val,
                                num_workers=self.num_workers, 
                                drop_last=False)


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


    
    # dataset = SimpleDataset(
    #     args.PREPROCESSED,
    #     args.WAV,
    #     preprocess_function=simple_audio_preprocess(args.SR,
    #                                                 2 * args.N_SIGNAL),
    #     split_set="full",
    #     transforms=Compose([
    #         RandomCrop(args.N_SIGNAL),
    #         RandomApply(
    #             lambda x: random_phase_mangle(x, 20, 2000, .99, args.SR),
    #             p=.8,
    #         ),
    #         Dequantize(16),
    #         lambda x: x.astype(np.float32),
    #     ]),
    # )

    # val = max((2 * len(dataset)) // 100, 1)
    # train = len(dataset) - val
    # train, val = random_split(
    #     dataset,
    #     [train, val],
    #     generator=torch.Generator().manual_seed(42),
    # )

    # train = DataLoader(train, args.BATCH, True, drop_last=True, num_workers=8)
    # val = DataLoader(val, args.BATCH, False, num_workers=8)