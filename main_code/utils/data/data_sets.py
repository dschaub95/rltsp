import numpy as np
import os
import json
from torch.utils.data import Dataset

from main_code.utils.data.tsp_transformer import (
    TSPEucTransformer,
    RandomTSPEucTransformation,
)


class TSPSample:
    def __init__(
        self, node_feats, opt_tour=None, opt_tour_len=None, edge_feats=None
    ) -> None:
        self.node_feats = node_feats
        self.opt_tour = opt_tour
        self.opt_tour_len = opt_tour_len
        self.edge_feats = edge_feats


class Generator:
    def __init__(self) -> None:
        self.num_samples = 0

    def get_new_instance(self, index):
        raise NotImplementedError


class DiskGenerator(Generator):
    """
    Loads tsp instances from disk on the fly
    """

    def __init__(self, dataset_path) -> None:
        super().__init__()
        self.path = dataset_path
        self.loaded_instances = 0
        self.num_samples = len(os.listdir(f"{self.path}"))

    def get_new_instance(self, index):
        idx_str = f"{index}".zfill(len(str(self.num_samples)))
        instance_path = f"{self.path}/tsp_{idx_str}"
        # load txt files directly as numpy arrays
        node_feats = np.loadtxt(f"{instance_path}/node_feats.txt")
        # used later with agnn
        # edge_weights = np.loadtxt(f'{instance_path}/edge_weights.txt')
        # edge_probs = np.loadtxt(f'{instance_path}/edge_probs.txt')
        with open(f"{instance_path}/solution.json", "r") as f:
            sol_dict = json.load(f)
        opt_tour = np.array(sol_dict["opt_tour"])
        opt_tour_len = sol_dict["opt_tour_length"]
        return node_feats, opt_tour_len


class RandomGenerator(Generator):
    """
    Generates random tsp instance
    """

    def __init__(self, num_samples, num_nodes, fixed_seed=False) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_samples = num_samples
        if fixed_seed:
            # every data set object generates the same instances
            # use for test set
            self.rng = np.random.default_rng(seed=37)
        else:
            # every data set object generates completely different instances
            # should be reproducible accross training runs
            # generate random seed for this run based on the global random seed
            seed = np.random.randint(2**63)
            self.rng = np.random.default_rng(seed=seed)

    def get_new_instance(self, index):
        return self.rng.random((self.num_nodes, 2))


class TSPDataset(Dataset):
    def __init__(self, generator) -> None:
        super().__init__()
        self.generator = generator
        self.num_samples = generator.num_samples

    def _get_new_instance(self, index):
        return self.generator.get_new_instance(index)

    def __getitem__(self, index):
        data = self._get_new_instance(index)
        return data

    def __len__(self):
        return int(self.num_samples)


class TSPTrainSet(TSPDataset):
    def __init__(self, generator, transform=None) -> None:
        super().__init__(generator)
        self.transform = transform
        # also add probability to transformations
        self.random_transformer = RandomTSPEucTransformation(pomo_first=False)

    def __getitem__(self, index):
        data = self._get_new_instance(index)
        if self.transform:
            data = self.random_transformer(data)
        return data


class TSPTestSet(TSPDataset):
    def __init__(self, generator, use_pomo_aug=False, sampling_steps=1) -> None:
        super().__init__(generator)
        self.use_pomo_aug = use_pomo_aug
        assert sampling_steps > 0
        self.sampling_steps = sampling_steps
        self.transformer = TSPEucTransformer()
        # specify which random transformations are allowed
        self.random_transformer = RandomTSPEucTransformation(pomo_first=True)
        self.last_orig_problem = None
        self.num_samples = self._adapt_sample_num(self.num_samples)

    def _adapt_sample_num(self, num_samples):
        if self.use_pomo_aug:
            self.sampling_steps = 8
            num_samples = num_samples * 8
        elif self.sampling_steps > 1:
            num_samples = num_samples * self.sampling_steps
        return num_samples

    def __getitem__(self, index):
        step = index % self.sampling_steps
        true_index = int((index - step) / self.sampling_steps)
        if step == 0:
            raw_data = self._get_new_instance(true_index)
            # convert potential tuple to list or put array into list
            if type(raw_data) is not np.ndarray:
                node_xy_data = raw_data[0]
                self.last_opt_len = raw_data[1]
            else:
                node_xy_data = raw_data
                self.last_opt_len = np.nan
            self.last_orig_problem = node_xy_data
            # reset transformer for sampling
            if self.sampling_steps > 1:
                self.random_transformer.reset()
        elif self.use_pomo_aug:
            node_xy_data = self.transformer.pomo_transform(
                self.last_orig_problem, variant=step
            )
        else:
            node_xy_data = self.random_transformer(self.last_orig_problem)
        # if we use augmentation it is relevant that we keep track of the original sample
        # and only generate a new one if we generated sampling steps - 1 many
        return node_xy_data, self.last_opt_len


class RandomTSPTestSet(TSPTestSet):
    def __init__(
        self, num_samples, num_nodes, use_pomo_aug=False, sampling_steps=1
    ) -> None:
        generator = RandomGenerator(num_samples, num_nodes, fixed_seed=True)
        super().__init__(
            generator, use_pomo_aug=use_pomo_aug, sampling_steps=sampling_steps
        )


class DiskTSPTestSet(TSPTestSet):
    def __init__(self, test_set_path, use_pomo_aug=False, sampling_steps=1) -> None:
        generator = DiskGenerator(test_set_path)
        super().__init__(
            generator, use_pomo_aug=use_pomo_aug, sampling_steps=sampling_steps
        )


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


class TSPDataSetRandomOrig(Dataset):
    def __init__(self, num_samples, num_nodes):
        self.num_samples = num_samples
        self.num_nodes = num_nodes

    def __getitem__(self, index):
        node_xy_data = np.random.rand(self.num_nodes, 2)

        return node_xy_data

    def __len__(self):
        return self.num_samples
