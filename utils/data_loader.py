import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.torch_objects import Tensor

def TSP_DATA_LOADER__RANDOM(num_samples, num_nodes, batch_size):
    dataset = TSPDataSetRandom(num_samples=int(num_samples), num_nodes=int(num_nodes))
    data_loader = DataLoader(dataset=dataset,
                             batch_size=int(batch_size),
                             shuffle=False,
                             num_workers=0,
                             collate_fn=TSP_collate_fn)
    return data_loader

class TSPDataSetRandom(Dataset):
    """
    Base randomly generated data set for train and test
    """
    def __init__(self, num_samples, num_nodes, fixed_seed=False):
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        if fixed_seed:
            # every data set object generates the same instances
            self.rng = np.random.default_rng(seed=37)
        else:
            # every data set object generates completely different instances
            self.rng = np.random.default_rng()

    def __getitem__(self, index):
        node_xy_data = self.rng.random((self.num_nodes, 2))
        # node_xy_data = np.random.rand(self.num_nodes, 2)
        return node_xy_data

    def __len__(self):
        return self.num_samples

class TSPTrainSetRandom(TSPDataSetRandom):
    def __init__(self, num_samples, num_nodes, clustered=False):
        super().__init__(num_samples, num_nodes, fixed_seed=False)
        self.clustered = clustered

class TSPTestSetRandom(TSPDataSetRandom):
    def __init__(self, num_samples, num_nodes, use_pomo_augmentation=False, sampling_steps=1):
        # fixed seed ensures each test set is the same
        super().__init__(num_samples, num_nodes, fixed_seed=True)
        self.last_orig_problem = None
        self.sampling_steps = sampling_steps
        self.use_pomo_augmentation = use_pomo_augmentation
    
    def __getitem__(self, index):
        if index == 0:
            node_xy_data = np.random.rand(self.num_nodes, 2)
            self.last_orig_problem = node_xy_data
        if self.use_pomo_augmentation:
            pass
        elif self.sampling_steps > 1:
            pass
        if index % self.sampling_steps:
            pass
        # if we use augmentation it is relevant that we keep track of the original sample 
        # and only generate a new one if we generated sampling steps - 1 many

        return node_xy_data


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
        super().__init__(dataset=TSPDataSetRandom(num_samples=num_samples, num_nodes=num_nodes), 
                         batch_size=batch_size, shuffle=False, 
                         num_workers=0, 
                         collate_fn=TSP_collate_fn)

class AugmentedTSPLoader(DataLoader):
    pass

