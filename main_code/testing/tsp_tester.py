import numpy as np
import time
import torch
import wandb
import pandas as pd
import json
from main_code.utils.torch_objects import device, Tensor, LongTensor
from main_code.environment.environment_new import GroupEnvironment
from main_code.utils.utils import AverageMeter
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
        logger=None,
        num_trajectories=1,
        num_nodes=100,
        num_samples=10000,
        sampling_steps=1,
        use_pomo_aug=False,
        test_set_path=None,
        test_batch_size=1024,
        log_period_sec=5,
        num_workers=4,
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
        self.num_workers = num_workers
        self._prepare_test_set(num_nodes, num_samples)
        # clip number of trajectories
        self.num_trajectories = num_trajectories

        # implement later
        wandb.define_metric("episode")
        wandb.define_metric("avg_error", step_metric="episode")

    def _prepare_test_set(self, num_nodes, num_samples):
        if self.test_set_path is not None:
            self.data_loader = DiskTSPTestDataLoader(
                self.test_set_path,
                self.test_batch_size,
                self.use_pomo_aug,
                self.sampling_steps,
                self.num_workers,
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

    def test(self, agent):
        self.result = TSPTestResult()
        agent.eval()
        # init this every time
        eval_dist_AM_0 = AverageMeter()
        if self.logger is not None:
            self.logger.info(
                "==================================================================="
            )
            self.logger.info(
                f"Max number of considered trajectories: {self.num_trajectories}"
            )
            self.logger.info(f"Number of sampling steps: {self.sampling_steps}")
            self.logger.info(f"Using POMO augmentation: {self.use_pomo_aug}")
            self.logger.logger_start = time.time()
        global_timer_start = time.time()
        episode = 0
        for batch_idx, (node_batch, opt_lens) in enumerate(self.data_loader):
            # print(opt_lens)
            batch = Tensor(node_batch)
            batch_s = batch.size(0)
            num_nodes = batch.size(1)
            num_trajectories = np.clip(self.num_trajectories, 1, num_nodes)
            episode += batch_s / self.sampling_steps
            with torch.no_grad():
                env = GroupEnvironment(batch, num_nodes)
                group_s = num_trajectories
                # get initial group state with specific nodes selected as first nodes in different tours
                group_state, reward, done = env.reset(group_size=group_s)
                # do the encoding only once
                agent.reset(group_state)
                while not done:
                    action, action_info = agent.get_action(group_state)
                    # shape = (batch, group)
                    group_state, reward, done = env.step(action)
            self._aggregate_results(
                reward,
                opt_lens,
                group_state,
                eval_dist_AM_0,
                global_timer_start,
                num_nodes,
                num_trajectories,
            )
            # do the intermediate logging
            # maybe also log action info
            self._log_intermediate(episode)
        self._log_final()
        return self.result

    def _aggregate_results(
        self,
        reward,
        opt_lens,
        group_state,
        eval_dist_AM_0,
        global_timer_start,
        num_nodes,
        num_trajectories,
    ):
        tours = group_state.selected_node_list
        # handle augmentation
        if self.use_pomo_aug or self.sampling_steps > 1:
            # reshape result reduce to most promising trajectories for each sampled graph
            # in case of sampling the final solution must be calculated as the true best solution with respect to the original problem (use group state)
            reward_sampling = get_group_travel_distances_sampling(
                tours, group_state.data, self.sampling_steps
            )
            # print(torch.allclose(reward_sampling, reward))
            reward_sampling = torch.reshape(
                reward_sampling, (-1, self.sampling_steps, num_trajectories)
            )
            reward, max_sample_indices = reward_sampling.max(dim=1)
            # reduce tours based on sampling steps
            tours = tours.reshape(
                (-1, self.sampling_steps, num_trajectories, num_nodes)
            )
            indices = max_sample_indices[:, None, :, None].expand(
                (-1, 1, -1, num_nodes)
            )
            tours = tours.gather(dim=1, index=indices).squeeze(dim=1)

        # max over trajectories
        max_reward, max_traj_indices = reward.max(dim=-1)
        eval_dist_AM_0.push(-max_reward)  # reward was given as negative dist

        # make final tour selection based on trajectories
        indices = max_traj_indices[:, None, None].expand(-1, 1, num_nodes)
        final_tours = tours.gather(dim=1, index=indices).squeeze(dim=1)

        # in case of sampling only select the optimal lengths for the max values
        opt_lens = opt_lens.reshape(-1, self.sampling_steps)[:, 0]

        # save all data into result object
        self.result.pred_lengths.extend((-max_reward).tolist())
        self.result.opt_lengths.extend(opt_lens.tolist())
        self.result.approx_errors = (
            (np.array(self.result.pred_lengths) / np.array(self.result.opt_lengths) - 1)
            * 100
        ).tolist()
        self.result.avg_approx_error = np.mean(self.result.approx_errors)
        self.result.avg_length = eval_dist_AM_0.peek()
        self.result.tours.extend(final_tours.tolist())
        self.result.global_computation_time = time.time() - global_timer_start

    def _log_intermediate(self, episode):
        # maybe also log action info
        if self.logger is not None:
            wandb.log(
                {
                    "avg_error": self.result.avg_approx_error,
                    "episode": episode,
                    "computation_time": self.result.global_computation_time,
                }
            )
            timestr = time.strftime(
                "%H:%M:%S", time.gmtime(self.result.global_computation_time)
            )
            percent = np.round((episode / self.num_samples) * 100, 1)
            episode_str = f"{int(episode)}".zfill(len(str(int(self.num_samples))))
            avg_length = np.round(self.result.avg_length, 7)
            avg_error = np.round(self.result.avg_approx_error, 7)
            log_str = f"Ep:{episode_str} ({percent:5}%)  T:{timestr}  avg.dist:{avg_length}  avg.error:{avg_error}%"
            self.logger.info(log_str)
            self.logger.logger_start = time.time()

    def _log_final(self):
        if self.logger is not None:
            log_data = {
                "sample_id": np.arange(len(self.result.approx_errors)),
                "approx_errors": self.result.approx_errors,
                "pred_lengths": self.result.pred_lengths,
            }
            log_tbl = wandb.Table(data=pd.DataFrame(log_data))
            wandb.log({"run_metrics": log_tbl})
            # also upload found tours
            max_tour_length = len(max(self.result.tours, key=len))
            tmp_array = np.full((max_tour_length,), np.nan)
            tour_data = {
                f"Instance {i}": np.concatenate(
                    (tour, tmp_array[0 : max_tour_length - len(tour)])
                )
                for i, tour in enumerate(self.result.tours)
            }
            tour_tbl = wandb.Table(data=pd.DataFrame(tour_data, copy=False))
            # make more efficient
            # min_tour_length = len(min(self.result.tours, key=len))
            # if min_tour_length < max_tour_length:
            #     tmp_array = np.full((max_tour_length,), np.nan)
            #     tour_array = np.array(
            #         [
            #             np.concatenate(
            #                 (tour, tmp_array[0 : max_tour_length - len(tour)])
            #             )
            #             for tour in self.result.tours
            #         ]
            #     ).transpose()
            # else:
            #     tour_array = np.array(self.result.tours).transpose()
            # tour_names = [f"Instance {i}" for i in range(self.num_samples)]
            # tour_indices = np.arange(max_tour_length)
            # tour_tbl = wandb.Table(
            #     data=pd.DataFrame(
            #         data=tour_array, columns=tour_names, index=tour_indices
            #     )
            # )
            wandb.log({"tours": tour_tbl})

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
        self.global_computation_time = 0
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
