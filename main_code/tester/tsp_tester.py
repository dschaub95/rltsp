import numpy as np
import time
import torch
import pandas as pd
import json
from main_code.utils.torch_objects import device, Tensor, LongTensor
from main_code.environment.environment_new import GroupEnvironment
from main_code.utils.utils import Average_Meter
from main_code.utils.data.tsp_transformer import get_group_travel_distances_sampling
from main_code.utils.data.data_loader import (
    RandomTSPTestDataLoader,
    DiskTSPTestDataLoader,
)


class BaseTester:
    pass


class TSPTester:
    """
    Provides a highlevel testing API, which executes a specified testing protocoll
    """

    def __init__(
        self,
        logger,
        num_trajectories=1,
        num_nodes=100,
        num_samples=10000,
        sampling_steps=1,
        use_pomo_aug=False,
        test_set_path=None,
        test_batch_size=1024,
        log_period_sec=5,
    ) -> None:
        # only relevant when generating test data on the fly
        self.logger = logger
        self.log_period_sec = log_period_sec
        self.sampling_steps = sampling_steps
        self.use_pomo_aug = use_pomo_aug
        # include for episode update
        if self.use_pomo_aug:
            self.sampling_steps = 8
        self.test_batch_size = test_batch_size
        self.test_set_path = test_set_path
        self._prepare_test_set(num_nodes, num_samples)
        # clip number of trajectories
        self.num_trajectories = np.clip(num_trajectories, 1, self.num_nodes)

        # implement later
        # self.env = GroupEnvironment()
        self.result = TSPTestResult()

    def _prepare_test_set(self, num_nodes, num_samples):
        if self.test_set_path is not None:
            self.data_loader = DiskTSPTestDataLoader(
                self.test_set_path,
                self.test_batch_size,
                self.use_pomo_aug,
                self.sampling_steps,
            )
        else:
            self.data_loader = RandomTSPTestDataLoader(
                num_samples,
                num_nodes,
                self.test_batch_size,
                self.use_pomo_aug,
                self.sampling_steps,
            )
        # let the data loader handle the sample and node num depending on the dataset etc
        self.num_samples = self.data_loader.num_samples
        self.num_nodes = self.data_loader.num_nodes

    def test(self, agent):
        agent.eval()

        # init this every time
        eval_dist_AM_0 = Average_Meter()

        self.logger.info(
            "==================================================================="
        )
        self.logger.info(f"Number of considered trajectories: {self.num_trajectories}")
        self.logger.info(f"Number of sampling steps: {self.sampling_steps}")
        self.logger.info(f"Using POMO augmentation: {self.use_pomo_aug}")

        timer_start = time.time()
        logger_start = time.time()
        episode = 0
        # for batch in self.data_loader:
        #     batch = Tensor(batch)
        #     batch_s = batch.size(0)
        #     episode += batch_s / self.sampling_steps
        for node_batch, opt_lens in self.data_loader:
            # print(opt_lens)
            batch = Tensor(node_batch)
            batch_s = batch.size(0)
            episode += batch_s / self.sampling_steps
            with torch.no_grad():
                env = GroupEnvironment(batch, self.num_nodes)
                group_s = self.num_trajectories
                # get initial group state with specific nodes selected as first nodes in different tours
                group_state, reward, done = env.reset(group_size=group_s)
                # do the encoding only once
                agent.reset(group_state)

                while not done:
                    action = agent.get_action(group_state)
                    # shape = (batch, group)
                    group_state, reward, done = env.step(action)

            tours = group_state.selected_node_list

            # handle augmentation
            if self.use_pomo_aug or self.sampling_steps > 1:
                # reshape result reduce to most promising trajectories for each sampled graph
                # in case of sampling the final solution must be calculated as the true best solution with respect to the original problem (use group state)
                reward_sampling = get_group_travel_distances_sampling(
                    env.group_state.selected_node_list, batch, self.sampling_steps
                )
                # print(torch.allclose(reward_sampling, reward))
                reward_sampling = torch.reshape(
                    reward_sampling, (-1, self.sampling_steps, self.num_trajectories)
                )
                reward, max_sample_indices = reward_sampling.max(dim=1)
                # reduce tours based on sampling steps
                tours = tours.reshape(
                    (-1, self.sampling_steps, self.num_trajectories, self.num_nodes)
                )
                indices = max_sample_indices[:, None, :, None].expand(
                    (-1, 1, -1, self.num_nodes)
                )
                tours = tours.gather(dim=1, index=indices).squeeze(dim=1)

            # max over trajectories
            max_reward, max_traj_indices = reward.max(dim=-1)
            eval_dist_AM_0.push(-max_reward)  # reward was given as negative dist

            # make final tour selection based on trajectories
            indices = max_traj_indices[:, None, None].expand(-1, 1, self.num_nodes)
            final_tours = tours.gather(dim=1, index=indices).squeeze(dim=1)

            # in case of sampling only select the optimal lengths for the max values
            opt_lens = opt_lens.reshape(-1, self.sampling_steps)[:, 0]

            # save all data into result object
            self.result.pred_lengths.extend((-max_reward).tolist())
            self.result.opt_lengths.extend(opt_lens.tolist())
            self.result.approx_errors = (
                (
                    np.array(self.result.pred_lengths)
                    / np.array(self.result.opt_lengths)
                    - 1
                )
                * 100
            ).tolist()
            self.result.avg_approx_error = np.mean(self.result.approx_errors)
            self.result.avg_length = eval_dist_AM_0.peek()
            self.result.tours.extend(final_tours.tolist())
            # do the logging
            if (time.time() - logger_start > self.log_period_sec) or (
                episode >= self.num_samples
            ):
                timestr = time.strftime(
                    "%H:%M:%S", time.gmtime(time.time() - timer_start)
                )
                percent = np.round((episode / self.num_samples) * 100, 1)
                episode_str = f"{int(episode)}".zfill(len(str(int(self.num_samples))))
                avg_length = np.round(self.result.avg_length, 7)
                avg_error = np.round(self.result.avg_approx_error, 7)
                log_str = f"Ep:{episode_str} ({percent:5}%)  T:{timestr}  avg.dist:{avg_length}  avg.error:{avg_error}%"
                self.logger.info(log_str)
                logger_start = time.time()
        # add some more infos to the result object
        self.result.computation_time = timestr
        return self.result

    def save_results(self, file_path):
        # save results as dataframe with instance id, distance, (approx ratio) and computation time
        self.result.to_json(file_path)

    def visualize_results(self):
        # after running the test directly save a result plot (average, std, distribution)
        # probably import function from another module
        pass


# not sure if useful
class TSPTestResult:
    def __init__(self, test_config=None) -> None:
        self.avg_approx_error = np.nan
        self.avg_length = np.nan
        self.computation_time = ""
        self.pred_lengths = []
        self.opt_lengths = []
        self.approx_errors = []
        self.tours = []
        self.tsp_labels = []
        # extra parameters to

        # test set stats (at best extract from sub test config)
        if test_config is not None:
            self.config = test_config
            self.num_nodes = np.nan
            # retrieve test set name if test set path was set
            self.test_set_name = self.config.test_set_path.split("/")[-1]

    def to_dict(self):
        # add properties to attrib dict
        result_dict = self.__dict__.copy()
        return result_dict

    def to_df(self):
        return pd.DataFrame.from_dict(self.to_dict(), orient="index").transpose()

    def to_json(self, file_path):
        # add properties to attrib dict

        # write param dict to json file
        with open(file_path, "w") as outfile:
            json.dump(self.to_dict(), outfile, indent=2)
