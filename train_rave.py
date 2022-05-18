import torch

from rave.model import RAVE
from rave.core import search_for_run

import pytorch_lightning as pl
from os import environ, path

import GPUtil as gpu

from pytorch_lightning.utilities.cli import LightningCLI
from utils.cli import CLI
from data.vctk import VCTKDataModule

cli = CLI(
        model_class=RAVE, 
        datamodule_class=pl.LightningDataModule, #pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_overwrite=True,
        run = False,
    )
  
# ------------------ Device check ------------------

# CUDA = gpu.getAvailable(maxMemory=.05)
# VISIBLE_DEVICES = environ.get("CUDA_VISIBLE_DEVICES", "")

# if VISIBLE_DEVICES:
#     use_gpu = int(int(VISIBLE_DEVICES) >= 0)
# elif len(CUDA):
#     environ["CUDA_VISIBLE_DEVICES"] = str(CUDA[0])
#     use_gpu = 1
# elif torch.cuda.is_available():
#     print("Cuda is available but no fully free GPU found.")
#     print("Training may be slower due to concurrent processes.")
#     use_gpu = 1
# else:
#     print("No GPU found.")
#     use_gpu = 0

batch_size = cli.config["data"].as_dict()["init_args"]["batch_size"]

x = torch.zeros(batch_size, 2**14)
cli.model.validation_step(x, 0)

run = search_for_run(None)
# if run is not None:
#     step = torch.load(run, map_location='cpu')["global_step"]
#     trainer.fit_loop.epoch_loop._batches_that_stepped = step

cli.trainer.fit(cli.model, cli.datamodule) #, ckpt_path=run)