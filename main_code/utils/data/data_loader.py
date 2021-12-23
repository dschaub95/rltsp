import numpy as np
import fnmatch
import os
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from main_code.utils.torch_objects import Tensor
from main_code.utils.data.tsp_transformer import TSPEucTransformer, RandomTSPEucTransformation

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


class TSPDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def _get_new_instance(self):
        raise NotImplementedError
    
    def __getitem__(self, index):
        data = self.get_new_instance()
        return data

    def __len__(self):
        raise NotImplementedError

class TSPDataSetRandom(Dataset):
    """
    Base randomly generated data set for train and test
    """
    def __init__(self, num_samples, num_nodes, fixed_seed=False):
        self.num_samples = int(num_samples)
        self.num_nodes = int(num_nodes)
        if fixed_seed:
            # every data set object generates the same instances
            self.rng = np.random.default_rng(seed=37)
        else:
            # every data set object generates completely different instances
            # should be reproducible accross training runs
            # generate random seed for this run
            seed = np.random.randint(2**63)
            self.rng = np.random.default_rng(seed=seed)
    
    def get_new_instance(self):
        return self.rng.random((self.num_nodes, 2))

    def __getitem__(self, index):
        node_xy_data = self.get_new_instance()
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
        self.num_samples = num_samples
        self.use_pomo_augmentation = use_pomo_augmentation
        assert sampling_steps > 0
        self.sampling_steps = sampling_steps
        self.transformer = TSPEucTransformer()
        # specify which random transformations are allowed
        self.random_transformer = RandomTSPEucTransformation(pomo_first=True)
        if self.use_pomo_augmentation:
            self.sampling_steps = 8
            self.num_samples = self.num_samples * 8
        elif self.sampling_steps > 1:
            self.num_samples = self.num_samples * self.sampling_steps
        # the augmented data set contains x times more samples
        super().__init__(self.num_samples, num_nodes, fixed_seed=True)
        self.last_orig_problem = None
    
    def __getitem__(self, index):
        step = index % self.sampling_steps
        if step == 0:
            node_xy_data = self.rng.random((self.num_nodes, 2))
            self.last_orig_problem = node_xy_data
            # reset transformer for sampling
            if self.sampling_steps > 1:
                self.random_transformer.reset()
        elif self.use_pomo_augmentation:
            node_xy_data = self.transformer.pomo_transform(self.last_orig_problem, variant=step)
        else:
            node_xy_data = self.random_transformer(self.last_orig_problem)
        # if we use augmentation it is relevant that we keep track of the original sample 
        # and only generate a new one if we generated sampling steps - 1 many
        return node_xy_data



class TSPDatasetSaved(Dataset):
    def __init__(self, path):
        # path to the dataset
        self.path = path
        self.cur_instance = 0

    def get_new_instance(self, index):
        # load problem instance in order
        # load node coordinates from file
        # load solution
        # load length
        # load heatmap
        pass 

    def __getitem__(self, index):
        node_xy_data = self.get_new_instance(self)
        self.cur_instance += 1

        return node_xy_data

    def __len__(self):
        # return number of files in the directory
        # return len(fnmatch.filter(os.listdir(self.path), '*.tsp'))
        return len(os.listdir(f'{self.path}/problems'))
        # return len(glob.glob(f'{self.path}/problems/*.tsp'))



