import numpy as np
import fnmatch
import os
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from main_code.utils.torch_objects import Tensor
from main_code.utils.data.tsp_transformer import TSPEucTransformer, RandomTSPEucTransformation
from main_code.utils.data.data_sets import TSPDataSetRandom, TSPTestSetRandom

def TSP_DATA_LOADER__RANDOM(num_samples, num_nodes, batch_size):
    dataset = TSPDataSetRandom(num_samples=int(num_samples), num_nodes=int(num_nodes))
    data_loader = DataLoader(dataset=dataset,
                             batch_size=int(batch_size),
                             shuffle=False,
                             num_workers=0,
                             collate_fn=TSP_collate_fn)
    return data_loader

class TSPDataLoader(DataLoader):
    """
    Needs suitable batchsize if augmentation is used in the data set
    """
    def __init__(self, dataset, batch_size, num_workers=0, shuffle=False):
        super().__init__(dataset=dataset,
                         batch_size=int(batch_size),
                         shuffle=shuffle,
                         num_workers=int(num_workers),
                         collate_fn=TSP_collate_fn)

class TSPTestDataLoaderRandom(TSPDataLoader):
    def __init__(self, num_samples, num_nodes, batch_size, use_pomo_augmentation=False, sampling_steps=1, num_workers=0, shuffle=False):
        dataset = TSPTestSetRandom(num_samples, num_nodes, use_pomo_augmentation, sampling_steps)
        # modify batch size in case augmentation is used, such that augmented samples fit into the same batch
        if use_pomo_augmentation:
            batch_size = np.floor(batch_size / 8) * 8 
        elif sampling_steps > 1:
            batch_size = np.floor(batch_size / sampling_steps) * sampling_steps
        super().__init__(dataset, batch_size, num_workers=num_workers, shuffle=shuffle)

def TSP_collate_fn(batch):
    batch = np.array(batch)
    node_xy = Tensor(batch)
    return node_xy

def identity_collate_fn(batch):
    return batch

def TSP_disk_collate_fn(batch):
    # batch consists of node coords, solution/length
    node_xy_batch = np.array([sample[0] for sample in batch])
    node_xy_batch = Tensor(node_xy_batch)
    solution_batch = np.array([sample[1] for sample in batch])
    solution_batch = Tensor(solution_batch)
    opt_len_batch = np.array([sample[2] for sample in batch])
    opt_len_batch = Tensor(opt_len_batch)
    # alternatively with np.array directly?
    return node_xy_batch, solution_batch, opt_len_batch

