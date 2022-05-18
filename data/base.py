
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from utils.data_utils import get_fragment_from_file, prepare_fn_groups_vocal, filter1_voice_wav
from utils.augmentations import aug

import pytorch_lightning as pl


class BaseDictDataset(Dataset):
    def __init__(self, 
                groups: dict, 
                nr_samples: int, 
                normalize: bool = True, 
                augmentations:dict = {}, 
                #augmentations_neg:dict = {}, 
                transform_override: bool = False,
                # positive_examples: str = 'same_clip',
                batch_sampling_mode: str = 'sample_clips',
                sr: int = 44100,
                multi_epoch: int = 1):
        self.groups = groups
        self.multi_epoch = multi_epoch
        self.nr_samples = nr_samples
        self.normalize = normalize
        self.transform_override = transform_override
        self.sr = sr

        self.augmentations = False
        if augmentations.get('enable', 0):
            if self.transform_override:
                raise ValueError("Transform override but augmentations passed are not the transforms")
            if augmentations['enable']:
                self.augmentations = augmentations
        if self.transform_override:
            self.augmentations = augmentations

        # self.augmentations_neg = False
        # if augmentations_neg.get('enable', 0):
        #     if augmentations_neg['enable']:
        #         self.augmentations_neg = augmentations_neg
            
        # self.positive_examples=positive_examples
        self.batch_sampling_mode=batch_sampling_mode

        self.groups_keys = list(self.groups.keys())
        self.inv_map = {fn: k for k, v in groups.items() for fn in v}
        self.inv_map_keys = list(self.inv_map.keys())
        self.inv_map_values = list(self.inv_map.values())

        if self.batch_sampling_mode == 'sample_clips':
            self.data_len = len(self.inv_map)
        else:
            self.data_len = len(self.groups)


    def getitem(self, item, file=None, group_name=None):
        raise NotImplementedError

    def get_fragment(self, fn):
        """
        Returns randomly sampled, normalized audio fragment from file fn of size self.nr_samples
        """
        frag = get_fragment_from_file(fn, self.nr_samples, self.normalize, draw_random=True, sr=self.sr)
        if frag is None: 
            print(f"Warning (get_fragment): could not get fragment from {fn}. Returning silence vector")
            frag = torch.zeros(self.nr_samples)
        return frag

    def get_clip_and_group_name(self, item):
        """
        Samples from dataset using batch_sampling_mode
        Returns:
            fn: samples filename
            group_name: group of name it belongs to 
        """

        if self.batch_sampling_mode == 'sample_clips':
            selec_fn = self.inv_map_keys[item]  # Sample a clip in the dataset
            group_name = self.inv_map_values[item]  # Gets clip artist
        elif self.batch_sampling_mode == 'sample_groups':
            group_name = self.groups_keys[item]  # Sample an artist (group)
        else:
            raise ValueError(f'Invalid batch sampling mode. Value was {self.batch_sampling_mode}')
        
        # Gets list of fns of the artist
        selec_group = self.groups[group_name]  

        fn = selec_fn if self.batch_sampling_mode == 'sample_clips' else selec_group[np.random.randint(len(selec_group))]
        return fn, group_name        
    
    def augment(self, data):
        """
        Performs augmentations descibred in dict self.augmentations on input data
        Can be overriden by transform_override, in which case  self.augmentations is
            expected to contain a dictionary whose values contains the transforms 
            themselves to be applied
        """
        
        override = False
        if self.transform_override:
            override = self.augmentations_pos
        data = aug(np.cast['float32'](data), self.augmentations, override=override)

        return data

    def __getitem__(self, item):
        item = item % self.data_len
        result = None
        while result is None:
            try:
                # Samples first filename and group_name
                fn, group_name = self.get_clip_and_group_name(item)
                # Additional get item
                result = self.getitem(item, file=fn, group_name=group_name)
                # Augments data
                result = self.augment(result)
                return result
            except AssertionError as e:
                raise e
            except Exception as e:
                raise e

    def __len__(self):
        return self.data_len * self.multi_epoch


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, 
                 dataset_dirs: list = [],
                 batch_size: int = 32,
                 batch_size_val: int = 32,
                 nr_samples: int = 176000,
                 normalize: bool = True,
                 num_workers: int = 8,
                 sr: int = 44100,
                #  positive_examples: str = "same_clip",
                 batch_sampling_mode: str = "sample_clips",
                 eval_frac: float = 0.1,
                 group_name_is_folder: bool = False,
                 group_by_artist : bool = False,
                 augs: dict = {},
                #  augs_neg: dict = {},
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
        # self.positive_examples = positive_examples
        self.batch_sampling_mode = batch_sampling_mode
        self.group_name_is_folder = group_name_is_folder
        self.group_by_artist = group_by_artist
        self.eval_frac = eval_frac
        self.groups = {}
        self.group_names = []  # maybe to register buffer
        self.augs = augs
        # self.augs_neg = augs_neg
        self.transform_override = transform_override
        self.nr_samples = nr_samples
        self.num_workers = num_workers
        self.verbose = verbose
        self.sr = sr
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
                    # select_only_groups=['out_44100'],
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
        
        self.prepare_data_end()

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

        self.setup_end()
        

    def prepare_data_end(self):
        return
    
    def setup_end(self):
        return
