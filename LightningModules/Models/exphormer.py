import torch.nn as nn
import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..gnn_base import GNNBase
from .multi_model import MultiModel

class Exphormer(GNNBase):
    def __init__(self, hparams):
        super().__init__(hparams)

        # Construct architecture
        # -------------------------
        input_dim = len(self.hparams["feature_set"])
        output_dim = self.hparams["nb_classes"]
        # Encode input features to hidden features
        self.model = MultiModel(input_dim, output_dim, hparams)

    def forward(self, batch):
        
        return self.model.forward(batch)
