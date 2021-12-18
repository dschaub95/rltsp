import numpy as np
import random
import time
import torch
from utils.torch_objects import device, Tensor, LongTensor
from environment.environment import GroupEnvironment
from utils.utils import Average_Meter
from utils.tsp_transformer import get_group_travel_distances_sampling
from utils.data_loader import TSPTestDataLoaderRandom

class TSPTester:
    """
    Provides a fixed and highlevel testing environment, which executes a specified testing protocoll
    """
    def __init__(self, 
                 config,
                 logger,
                 num_trajectories=1,                 
                 num_nodes=20,
                 num_samples=1e+4, 
                 sampling_steps=1, 
                 use_pomo_augmentation=False,
                 test_set_path=None,
                 test_batch_size=1024) -> None:
        # only relevant when generating test data on the fly
        self.cfg = config
        self.logger = logger
        self.num_trajectories = np.clip(num_trajectories, 1, num_nodes)
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.sampling_steps = sampling_steps
        self.use_pomo_augmentation = use_pomo_augmentation
        # include for episode update
        if self.use_pomo_augmentation:
            self.sampling_steps = 8
    
        if test_batch_size is None:
            self.test_batch_size = self.cfg.TEST_BATCH_SIZE
        else:
            self.test_batch_size = test_batch_size
        if test_set_path is not None:
            pass
        else:
            self.data_loader = TSPTestDataLoaderRandom(num_samples, num_nodes, self.test_batch_size, self.use_pomo_augmentation, self.sampling_steps)
        # implement later
        # self.env = GroupEnvironment()

    def test(self, model):
        model.eval()
        # iterate over specified testsets or rather testset dataloaders
        
        # init this every time
        eval_dist_AM_0 = Average_Meter()

        self.logger.info('===================================================================')
        self.logger.info(f'Number of considered trajectories: {self.num_trajectories}')
        self.logger.info(f'Number of sampling steps: {self.sampling_steps}')
        self.logger.info(f'Using POMO augmentation: {self.use_pomo_augmentation}')

        timer_start = time.time()
        logger_start = time.time()
        episode = 0
        for batch in self.data_loader:
            batch_s = batch.size(0)
            episode += batch_s / self.sampling_steps
            with torch.no_grad():
                
                env = GroupEnvironment(batch, self.num_nodes)
                group_s = self.num_trajectories
                group_state, reward, done = env.reset(group_size=group_s)
                model.reset(group_state)

                # First Move is given
                first_action = LongTensor(np.arange(group_s))[None, :].expand(batch_s, group_s)
                group_state, reward, done = env.step(first_action)

                while not done:
                    model.update(group_state)
                    action_probs = model.get_action_probabilities()
                    # shape = (batch, group, TSP_SIZE)
                    action = action_probs.argmax(dim=2)
                    # shape = (batch, group)
                    group_state, reward, done = env.step(action)
            # handle augmentation
            if self.use_pomo_augmentation:
                # reshape result reduce to most promising trajectories for each sampled graph
                # we can use the original reward since the length of the tour is not affected by the pomo augmentation
                # reshaping depends on whether we use a torch tensor as original input or a numpy tensor
                reward = torch.reshape(reward, (-1, 8, self.num_trajectories))
                reward, _ = reward.max(dim=1)
            elif self.sampling_steps > 1:
                # reshape result reduce to most promising trajectories for each sampled graph
                # in case of sampling the final solution must be calculated as the true best solution with respect to the original problem (use group state)
                reward_sampling = get_group_travel_distances_sampling(env.group_state.selected_node_list, batch, self.sampling_steps)
                # print(torch.allclose(reward_sampling, reward))
                reward_sampling = torch.reshape(reward_sampling, (-1, self.sampling_steps, self.num_trajectories))
                reward, _ = reward_sampling.max(dim=1)
            # the max does not do anything if we only use one trajectory
            max_reward, _ = reward.max(dim=-1)
            eval_dist_AM_0.push(-max_reward)  # reward was given as negative dist
            # do the logging
            if (time.time()-logger_start > self.cfg.LOG_PERIOD_SEC) or (episode >= self.num_samples):
                timestr = time.strftime("%H:%M:%S", time.gmtime(time.time()-timer_start))
                percent = np.round((episode / self.num_samples) * 100, 1)
                log_str = f"Ep:{int(episode)} ({percent}%)  T:{timestr}  avg.dist:{eval_dist_AM_0.peek()}"
                self.logger.info(log_str)
                logger_start = time.time()
        # return average dist
        self.result = eval_dist_AM_0.peek()
        return self.result

    def save_results(self):
        # save results as dataframe with instance id, distance, (approx ratio) and computation time
        pass

    def visualize_results(self):
        # after running the test directly save a result plot (average, std, distribution)
        # probably import function from another module
        pass

# not sure if useful
class TestResult:
    def __init__(self) -> None:
        self.avg_dist = None
        self.distances = None
        self.computation_time = None