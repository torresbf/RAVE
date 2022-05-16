from pytorch_lightning.utilities.cli import LightningCLI
import torch

class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # parser.add_argument("--ckpt_path", default=None)
        parser.add_argument("--test", default=None)

        # parser.link_arguments("data.batch_size", "model.batch_size")

        # parser.link_arguments("model.embedding_dim", "model.output_dim",  apply_on='parse')
        # parser.link_arguments("model.embedding_dim", "model.encoder.embedding_dim")
        # parser.link_arguments("model.output_dim", "model.projection.output_dim")

        # parser.add_optimizer_args(torch.optim.Adam,
        #                           link_to="model.optimizer1_init")
