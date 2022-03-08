import numpy as np
import fnmatch
import os
import glob
import torch
from torch.utils import data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from main_code.utils.torch_objects import Tensor
from main_code.utils.data.tsp_transformer import (
    TSPEucTransformer,
    RandomTSPEucTransformation,
)
from main_code.utils.data.data_sets import (
    TSPDataSetRandom,
    TSPDataSetRandomOrig,
    RandomTSPTestSet,
    DiskTSPTestSet,
)


def TSP_DATA_LOADER__RANDOM(num_samples, num_nodes, batch_size):
    dataset = TSPDataSetRandom(num_samples=int(num_samples), num_nodes=int(num_nodes))
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=TSP_collate_fn,
    )
    return data_loader


class TSPTestDataLoader(DataLoader):
    """
    Needs suitable batchsize if augmentation is used in the data set
    """

    def __init__(
        self,
        dataset,
        batch_size,
        collate_fn,
        use_pomo_aug=False,
        sampling_steps=1,
        num_workers=8,
    ):
        # modify batch size in case augmentation is used, such that augmented samples all fit into the same batch
        if use_pomo_aug:
            batch_size = np.floor(batch_size / 8) * 8
        elif sampling_steps > 1:
            batch_size = np.floor(batch_size / sampling_steps) * sampling_steps
        if num_workers > 0:
            # torch.multiprocessing.set_start_method('spawn')
            pass
        self.num_samples = int(len(dataset) / sampling_steps)
        # num depends on the dataset especially in case of a disk dataset
        self.num_nodes = dataset[0][0].shape[0]
        super().__init__(
            dataset=dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            collate_fn=collate_fn,
        )


class RandomTSPTestDataLoader(TSPTestDataLoader):
    def __init__(
        self, num_samples, num_nodes, batch_size, use_pomo_aug=False, sampling_steps=1
    ):
        self.test_set = RandomTSPTestSet(
            num_samples, num_nodes, use_pomo_aug, sampling_steps
        )
        super().__init__(
            self.test_set,
            batch_size,
            collate_fn=TSP_general_collate_fn,
            use_pomo_aug=use_pomo_aug,
            sampling_steps=sampling_steps,
        )


class DiskTSPTestDataLoader(TSPTestDataLoader):
    def __init__(
        self,
        test_set_path,
        batch_size,
        use_pomo_aug=False,
        sampling_steps=1,
        num_workers=0,
    ):
        # load data set from disk
        self.test_set = DiskTSPTestSet(test_set_path, use_pomo_aug, sampling_steps)
        super().__init__(
            self.test_set,
            batch_size,
            collate_fn=TSP_general_collate_fn,
            use_pomo_aug=use_pomo_aug,
            sampling_steps=sampling_steps,
            num_workers=num_workers,
        )


def TSP_collate_fn(batch):
    node_xy = np.array(batch)
    node_xy = Tensor(node_xy)
    return node_xy


def TSP_multi_collate_fn(batch):
    # batch consists of node coords, solution/length
    node_xy_batch = np.array([sample[0] for sample in batch])
    node_xy_batch = Tensor(node_xy_batch)
    solution_batch = np.array([sample[1] for sample in batch])
    solution_batch = Tensor(solution_batch)
    opt_len_batch = np.array([sample[2] for sample in batch])
    opt_len_batch = Tensor(opt_len_batch)
    # alternatively with np.array directly?
    return node_xy_batch, solution_batch, opt_len_batch


def TSP_general_collate_fn(batch):
    tmp_arr = np.array(batch, dtype=object)
    batch_s = tmp_arr.shape[0]
    num_entries = tmp_arr.shape[1]
    if num_entries > 1:
        num_feats = tmp_arr[0, 0].shape[1]
    else:
        num_feats = tmp_arr[0].shape[1]
    node_feats_batch = np.concatenate(tmp_arr[:, 0]).reshape(batch_s, -1, num_feats)
    # node_feats_batch = Tensor(node_feats_batch)
    # opt_tour_batch = np.concatenate(tmp_arr[:,1]).reshape(batch_s,-1).astype(np.int8)
    if num_entries > 1:
        opt_len_batch = tmp_arr[:, 1].astype(np.float64)
    else:
        opt_len_batch = np.empty((batch_s,)) * np.nan
    return node_feats_batch, opt_len_batch
