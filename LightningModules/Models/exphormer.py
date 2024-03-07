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
        batch.x = self.concat_feature_set(batch)
        # Add dummy edge_index
        batch.edge_index = torch.empty([2, 0], dtype=torch.long, device=batch.batch.device)
        pred = self.model.forward(batch)
        return pred
