import torch
from torch.utils.data import DataLoader, random_split, Dataset

from rave.model import RAVE
from rave.core import random_phase_mangle, EMAModelCheckPoint
from rave.core import search_for_run

from udls import SimpleDataset, simple_audio_preprocess
from effortless_config import Config, setting
import pytorch_lightning as pl
from os import environ, path
import numpy as np

import GPUtil as gpu

from udls.transforms import Compose, RandomApply, Dequantize, RandomCrop
from audiomentations import Gain

from data.data import DataModule

from pytorch_lightning.utilities.cli import LightningCLI


if __name__ == "__main__":

    class args(Config):
        groups = ["small", "large"]

        DATA_SIZE = 16
        CAPACITY = setting(default=64, small=32, large=64)
        LATENT_SIZE = 128
        BIAS = True
        NO_LATENCY = False
        RATIOS = setting(
            default=[4, 4, 4, 2],
            small=[4, 4, 4, 2],
            large=[4, 4, 2, 2, 2],
        )

        MIN_KL = 1e-1
        MAX_KL = 1e-1
        CROPPED_LATENT_SIZE = 0
        FEATURE_MATCH = True

        LOUD_STRIDE = 1

        USE_NOISE = True
        NOISE_RATIOS = [4, 4, 4]
        NOISE_BANDS = 5

        D_CAPACITY = 16
        D_MULTIPLIER = 4
        D_N_LAYERS = 4

        WARMUP = setting(default=1000000, small=1000000, large=3000000)
        MODE = "hinge"
        CKPT = None

        PREPROCESSED = None
        WAV = None
        SR = 48000
        N_SIGNAL = 65536
        MAX_STEPS = setting(default=3000000, small=3000000, large=6000000)
        VAL_EVERY = 10000

        BATCH = 8

        NAME = None

    args.parse_args()

    assert args.NAME is not None
    model = RAVE(
        data_size=args.DATA_SIZE,
        capacity=args.CAPACITY,
        latent_size=args.LATENT_SIZE,
        ratios=args.RATIOS,
        bias=args.BIAS,
        loud_stride=args.LOUD_STRIDE,
        use_noise=args.USE_NOISE,
        noise_ratios=args.NOISE_RATIOS,
        noise_bands=args.NOISE_BANDS,
        d_capacity=args.D_CAPACITY,
        d_multiplier=args.D_MULTIPLIER,
        d_n_layers=args.D_N_LAYERS,
        warmup=args.WARMUP,
        mode=args.MODE,
        no_latency=args.NO_LATENCY,
        sr=args.SR,
        min_kl=args.MIN_KL,
        max_kl=args.MAX_KL,
        cropped_latent_size=args.CROPPED_LATENT_SIZE,
        feature_match=args.FEATURE_MATCH,
    )

    # For every dataset (folder with group names in subfolders) do parse
    # -----------------------------------------
    nr_samples= 65536 # For 4s, nr_samples is sr*4
    normalize=True
    # augmentations_pos=augs_pos, 
    # augmentations_neg=augs_neg,
    augs_neg = {}
    augs_pos = {}
    positive_examples='same_clip'
    batch_size = 4
    batch_size_val = 2
    batch_sampling_mode='sample_clips'
    num_workers=30
    eval_frac = 0.1

    transforms=Compose([
            #RandomCrop(args.N_SIGNAL),
            RandomApply(
                lambda x: random_phase_mangle(x, 20, 2000, .99, args.SR),
                p=.8,
            ),
            Dequantize(16),
            lambda x: (Gain(min_gain_in_db=-6, max_gain_in_db=0, 
                            p=0.5)(x, sample_rate = args.SR)),
            lambda x: x.astype(np.float32),
        ])
    
    # -----------------------------------------

    x = torch.zeros(batch_size, 2**14)
    model.validation_step(x, 0)

    #--------------------

    datamodule = DataModule(dataset_dirs = ['/home/cyran/RAVE'],
                            batch_size = batch_size,
                            batch_size_val = batch_size_val,
                            nr_samples = nr_samples,
                            normalize = normalize,
                            num_workers = num_workers,
                            positive_examples = "same_clip",
                            batch_sampling_mode = "sample_clips",
                            eval_frac = eval_frac,
                            mask_samples = False,
                            augs_pos = {"custom_transform": transforms },
                            augs_neg = {},
                            transform_override = True,
                            verbose=True)

    # groups = {}
    # datasets = ['/home/cyran/RAVE']
    # for dataset in datasets:
    #     groups = prepare_fn_groups_vocal(
    #         dataset,
    #         groups=groups,
    #         select_only_groups=['test_data_dir'],
    #         filter_fun_level1=filter1_voice_wav,
    #         group_name_is_folder=True,
    #         group_by_artist=False)

    # inv_group = {fn: k for k, v in groups.items() for fn in v}

    # # Get group names
    # group_names = list(groups.keys())
    # np.random.shuffle(group_names)

    # n_files = sum([len(group) for group in list(groups.values())])
    # n_groups = len(group_names)

    # print(f'Number of files in dataset: {n_files}, split into {n_groups} artists')
    # #print(f"Augmentations on positive examples: {json.dumps(self.augs_pos, indent=4, sort_keys=True)}")
    # #print(f"Augmentations on negative examples: {json.dumps(self.augs_neg, indent=4, sort_keys=True)}")

    # count_elements_in_dict_split = lambda dic, keys_subset : sum([len(dic[key]) for key in keys_subset])

    # ## CHANGE THIS IS TEST CODE
    # eval_split = int(len(group_names) * eval_frac)
    # groups_train = dict((k, groups[k]) for k in group_names[eval_split:])
    # groups_eval = dict((k, groups[k]) for k in group_names[:eval_split])
    # n_files_train = count_elements_in_dict_split(groups, group_names[eval_split:])
    # n_files_eval = count_elements_in_dict_split(groups, group_names[:eval_split])

    # if batch_sampling_mode == 'sample_clips':
    #     n_batches = n_files_train / batch_size
    # elif batch_sampling_mode == 'sample_groups':
    #     n_batches = eval_split / batch_size
    # else:
    #     raise ValueError(f'Invalid batch sampling mode. Value was {batch_sampling_mode}')

    # print(f"Number of training batches: {n_batches}")
    # print(f"Size train (groups) : {n_groups-eval_split}, eval: {eval_split}")
    # print(f"Size train (files): {n_files_train}")
    # print(f"Size eval (files): {n_files_eval}")

    # train = DataLoader(TestLoader( groups_train, 
    #                                     nr_samples= nr_samples, # For 4s, nr_samples is sr*4
    #                                     normalize=normalize,
    #                                     augmentations_pos=augs_pos, 
    #                                     augmentations_neg=augs_neg,
    #                                     positive_examples=positive_examples,
    #                                     batch_sampling_mode=batch_sampling_mode,
    #                                     multi_epoch=1),
    #                         shuffle=True, 
    #                         batch_size=batch_size,
    #                         num_workers=num_workers, 
    #                         drop_last=True)

    
    # val = DataLoader(TestLoader(groups_eval,
    #                                 nr_samples=nr_samples, # For 4s, nr_samples is sr*4
    #                                 normalize=normalize,
    #                                 augmentations_pos={}, 
    #                                 augmentations_neg={},
    #                                 positive_examples=positive_examples,
    #                                 batch_sampling_mode=batch_sampling_mode,
    #                                 multi_epoch=1),
    #                     shuffle=False, 
    #                     batch_size=batch_size_val,
    #                     num_workers=num_workers, 
    #                     drop_last=True)


    # -----------------------------------------

    # CHECKPOINT CALLBACKS
    validation_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="validation",
        filename="best",
    )
    last_checkpoint = pl.callbacks.ModelCheckpoint(filename="last")

    CUDA = gpu.getAvailable(maxMemory=.05)
    VISIBLE_DEVICES = environ.get("CUDA_VISIBLE_DEVICES", "")

    if VISIBLE_DEVICES:
        use_gpu = int(int(VISIBLE_DEVICES) >= 0)
    elif len(CUDA):
        environ["CUDA_VISIBLE_DEVICES"] = str(CUDA[0])
        use_gpu = 1
    elif torch.cuda.is_available():
        print("Cuda is available but no fully free GPU found.")
        print("Training may be slower due to concurrent processes.")
        use_gpu = 1
    else:
        print("No GPU found.")
        use_gpu = 0

    use_gpu = 1

    val_check = {}
    # if len(datamodule.train_dataloader()) >= args.VAL_EVERY:
    #     val_check["val_check_interval"] = args.VAL_EVERY
    # else:
    #     nepoch = args.VAL_EVERY // len(datamodule.train_dataloader())
    #     val_check["check_val_every_n_epoch"] = nepoch

    val_check["val_check_interval"] = args.VAL_EVERY
    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(path.join("runs", args.NAME),
                                            name="rave"),
        gpus=use_gpu,
        callbacks=[validation_checkpoint, last_checkpoint],
        max_epochs=100000,
        max_steps=args.MAX_STEPS,
        **val_check,
    )

    run = search_for_run(args.CKPT)
    if run is not None:
        step = torch.load(run, map_location='cpu')["global_step"]
        trainer.fit_loop.epoch_loop._batches_that_stepped = step

    trainer.fit(model, datamodule, ckpt_path=run)