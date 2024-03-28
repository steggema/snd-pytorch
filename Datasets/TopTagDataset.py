import os

from typing import List

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import download_url

default_feature_scales = {
    'jet_pt': 1000.,
    'jet_px': 1000.,
    'jet_py': 1000.,
    'jet_pz': 1000.,
    'jet_pE': 1000.,
    'jet_mass': 100.,
    'jet_phi': 3.14159,
    'jet_eta': 3.14159,
    'pE': 100.,
    'px': 100.,
    'py': 100.,
    'pz': 100.,
    'log_delta_E': 10.,
    'log_delta_pt': 10.,   
    'log_E': 10.,
    'log_pt': 10.,
}

class TopTagDataset(InMemoryDataset):
    def __init__(self, root, feature_scales=None, transform=None, pre_transform=None, pre_filter=None):
        self.feature_scales = default_feature_scales if feature_scales is None else feature_scales
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            os.path.join(self.raw_dir, 'train.h5'), 
            os.path.join(self.raw_dir, 'test.h5'), 
            os.path.join(self.raw_dir, 'val.h5')
            ]

    @property
    def processed_file_names(self) -> List[str]:
        return [
            os.path.join(self.root, 'train.pt'),
            os.path.join(self.root, 'test.pt'),
            os.path.join(self.root, 'val.pt')
        ]

    def download(self) -> None:
        # Download to `self.raw_dir`.
        urls = [
            'https://zenodo.org/records/2603256/files/test.h5?download=1', 
            'https://zenodo.org/records/2603256/files/train.h5?download=1', 
            'https://zenodo.org/records/2603256/files/val.h5?download=1'
            ]
        for url in urls:
            download_url(url, self.raw_dir)

    def process(self) -> None:
        # Read data into huge `Data` list.

        for input_file, output_file in zip(self.raw_file_names, self.processed_file_names):
            data = pd.read_hdf(input_file, key='table')

            jet_batch = []
            for jet_id, jet_tuple in enumerate(tqdm(data.itertuples(), total=len(data))):
                jet_batch.append(self.hdf5_to_pyg_event((jet_tuple, jet_id), self.feature_scales))
            print(f"Saving to file {output_file}")
            self.save(jet_batch, output_file)


    def build_all_features(self, jet, feature_scales) -> Data:
        p = self.get_four_momenta(jet)
        y = torch.tensor(jet.is_signal_new)

        pt, delta_eta, delta_phi, jet_p4, jet_pt, jet_eta, jet_phi = self.get_higher_features(p)
        delta_pt = pt / jet_pt
        log_delta_pt = torch.log(delta_pt)
        delta_E = p[:, 0] / jet_p4[0]
        log_delta_E = torch.log(delta_E)
        delta_R = torch.sqrt( delta_eta**2 + delta_phi**2 )
        jet_mass = torch.sqrt(jet_p4[0]**2 - jet_p4[1]**2 - jet_p4[2]**2 - jet_p4[3]**2)

        x = torch.stack([
            p[:, 0]/feature_scales['pE'], 
            p[:, 1]/feature_scales['px'],
            p[:, 2]/feature_scales['py'],
            p[:, 3]/feature_scales['pz'],
            torch.log(pt)/feature_scales['log_pt'], 
            torch.log(p[:, 0])/feature_scales['log_E'], 
            delta_pt, 
            log_delta_pt/feature_scales['log_delta_pt'], 
            delta_E, 
            log_delta_E, 
            delta_R, 
            delta_eta, 
            delta_phi
            ], axis=1)

        pyg_jet = Data(x=x, y=y, num_nodes=len(delta_E),
                       jet_pt=jet_pt/feature_scales['jet_pt'],
                       jet_pE=jet_p4[0]/feature_scales['jet_pE'], 
                       jet_px=jet_p4[1]/feature_scales['jet_px'], 
                       jet_py=jet_p4[2]/feature_scales['jet_py'],
                       jet_pz=jet_p4[3]/feature_scales['jet_pz'],
                       jet_mass=jet_mass/feature_scales['jet_mass'],
                       jet_eta=jet_eta/feature_scales['jet_eta'], 
                       jet_phi=jet_phi/feature_scales['jet_phi'])

        # # Convert all to float
        # for key in pyg_jet.keys():
        #     pyg_jet[key] = pyg_jet[key].float()

        return pyg_jet

    def get_four_momenta(self, jet_tuple):
        # Set all values to zero
        energies = torch.tensor([getattr(jet_tuple, f'E_{i}') for i in range(200)])
        x_values = torch.tensor([getattr(jet_tuple, f'PX_{i}') for i in range(200)])
        y_values = torch.tensor([getattr(jet_tuple, f'PY_{i}') for i in range(200)])
        z_values = torch.tensor([getattr(jet_tuple, f'PZ_{i}') for i in range(200)])
        all_values = torch.stack([energies, x_values, y_values, z_values], dim=1)

        # Only keep existing jets
        existing_jet_mask = energies > 0
        return all_values[existing_jet_mask]

    def calc_kinematics(self, x, y, z):
        pt = np.sqrt(x**2 + y**2)
        theta = np.arctan2(pt, z)
        eta = -1. * np.log(np.tan(theta / 2.))
        phi = np.arctan2(y, x)
        
        return pt, eta, phi

    def get_higher_features(self, p):
        
        _, x, y, z = p.T
        pt, eta, phi = self.calc_kinematics(x,y,z)
        
        jet_p4 = p.sum(0)        
        jet_pt, jet_eta, jet_phi = self.calc_kinematics(jet_p4[1], jet_p4[2], jet_p4[3])
        
        delta_eta = eta - jet_eta
        delta_phi = phi - jet_phi
        delta_phi[delta_phi > np.pi] -= 2 * np.pi
        delta_phi[delta_phi < -np.pi] += 2 * np.pi
        
        return pt, delta_eta, delta_phi, jet_p4, jet_pt, jet_eta, jet_phi

    def hdf5_to_pyg_event(self, jet_entry, feature_scales) -> List[Data]:
        jet, _ = jet_entry

        pyg_jet = self.build_all_features(jet, feature_scales)
        
        return pyg_jet
    