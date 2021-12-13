import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.torch_objects import Tensor

def TSP_DATA_LOADER__RANDOM(num_samples, num_nodes, batch_size):
    dataset = TSP_Dataset_Random(num_samples=int(num_samples), num_nodes=int(num_nodes))
    data_loader = DataLoader(dataset=dataset,
                             batch_size=int(batch_size),
                             shuffle=False,
                             num_workers=0,
                             collate_fn=TSP_collate_fn)
    return data_loader

class TSP_Dataset_Random(Dataset):
    def __init__(self, num_samples, num_nodes):
        self.num_samples = num_samples
        self.num_nodes = num_nodes

    def __getitem__(self, index):
        node_xy_data = np.random.rand(self.num_nodes, 2)

        return node_xy_data

    def __len__(self):
        return self.num_samples


def TSP_collate_fn(batch):
    batch = np.array(batch)
    node_xy = Tensor(batch)
    return node_xy

class TSP_Dataset_from_disk(Dataset):
    def __init__(self, num_samples, num_nodes):
        self.num_samples = num_samples
        self.num_nodes = num_nodes

    def __getitem__(self, index):
        node_xy_data = np.random.rand(self.num_nodes, 2)

        return node_xy_data

    def __len__(self):
        return self.num_samples


def TSP_collate_fn(batch):
    batch = np.array(batch)
    node_xy = Tensor(batch)
    return node_xy

class TSPLoader(DataLoader):
    def __init__(self, data_set, num_samples, num_nodes, batch_size):
        super().__init__(dataset=TSP_Dataset_Random(num_samples=num_samples, num_nodes=num_nodes), 
                         batch_size=batch_size, shuffle=False, 
                         num_workers=0, 
                         collate_fn=TSP_collate_fn)

class AugmentedTSPLoader(DataLoader):
    pass

