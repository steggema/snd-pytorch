import os

from typing import List

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset


class SNDDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        # Maybe just listing the directories is more elegant...
        train_indices = [i for i in range(14)] + [i for i in range(20, 26)]
        return [os.path.join(self.root, f'train/output_{i}.pt') 
                for i in train_indices] + [os.path.join(self.root, f'test/output_{i}.pt') 
                for i in (17, 18, 19, 26, 27, 28, 29)] + [os.path.join(self.root, f'val/output_{i}.pt') for i in (14, 15, 16)]

    @property
    def processed_file_names(self) -> List[str]:
        return [
            os.path.join(self.root, 'train.pt'),
            os.path.join(self.root, 'test.pt'),
            os.path.join(self.root, 'val.pt')
        ]

    def process(self) -> None:
        # Read data into huge `Data` list.
        breakpoints = [20, 27, 30]
        j = 0
        batch = []
        for i, input_file in enumerate(tqdm(self.raw_file_names)):
            input = torch.load(input_file)
            for e in input:
                if len(e.vertical) == 0:
                    continue

                x = torch.stack([e.vertical, e.strip_x, e.strip_y, e.strip_z, e.strip_x_end, e.strip_y_end, e.strip_z_end, e.det]).T
                batch.append(Data(x=x, y=e.y, start_z=e.start_z, pz=e.pz))
            
            if i+1 >= breakpoints[j]:
                print(f"Saving to file {self.processed_file_names[j]}")
                self.save(batch, self.processed_file_names[j])
                j += 1
                batch.clear()
