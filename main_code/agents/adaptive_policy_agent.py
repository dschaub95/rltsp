from base64 import encode
from main_code.agents.base_agent import BaseAgent
from main_code.agents.policy_agent import PolicyAgent
from main_code.nets.pomo import PomoNetwork
from main_code.utils.utils import AverageMeter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from main_code.utils.torch_objects import Tensor, LongTensor, device
from main_code.environment.environment_new import (
    GroupEnvironment,
    GroupState,
    MultiGroupState,
)
import wandb
import copy


class ActionBuffer:
    def __init__(self) -> None:
        self.reset()

    def get_action(self):
        action = self.best_actions[:, self.counter, :]
        self.counter += 1
        return action

    # def __getitem__(self, index):
    #     # make actions easily acessible via index
    #     return self.best_actions[:, index, :]

    # def __len__(self):
    #     if self.best_actions is None:
    #         return 0
    #     else:
    #         return self.best_actions.size(1)

    # def add_action(self, action):
    #     # transform action into suitable shape
    #     action = action[:, None, :]
    #     # shape = (batch_s, num, trajects)
    #     # add action to memory
    #     if self.best_actions is None:
    #         self.best_actions = action
    #     else:
    #         self.best_actions = torch.cat([self.best_actions, action], dim=1)

    def _action_seq_to_tensor(self, action_seq):
        # make this more efficient potentially
        action_seq = [action[:, None, :] for action in action_seq]
        action_tensor = torch.cat(action_seq, dim=1)
        # action_tensor = torch.cat(action_seq).view(batch_s, -1, num_trajectories)
        return action_tensor

    def update(self, action_seq, lengths, indices=None):
        # update the memory if an action sequence for any sample is better
        # first check lengths
        action_tensor = self._action_seq_to_tensor(action_seq)
        if self.best_actions is None:
            self.best_actions = action_tensor
            self.best_lengths = lengths
        else:
            if indices is None:
                # directly compare lengths and current best lengths
                mask = lengths < self.best_lengths
                # mask = mask[:, None, None].expand()
                self.best_actions[mask, :, :] = action_tensor[mask, :, :]
                self.best_lengths[mask] = lengths[mask]
            else:
                # compare lengths by extracting lengths based on indices
                pass

    def reset(self):
        self.best_actions = None
        self.best_lengths = None
        self.counter = 0


# think about implementing external fine tuner
class AdaptivePolicyAgent(PolicyAgent):
    def __init__(
        self,
        policy_net,
        num_epochs=4,
        batch_size=8,
        lr_rate=1e-4,
        weight_decay=1e-6,
        lr_decay_epoch=1.0,
        lr_decay_gamma=1.0,
    ) -> None:
        # extra hyperpramaters for learning as mcts
        # specifically should have parameter defining the batch size for learning and the state space size
        # and standard learning parameters
        # data loader should return embedded nodes and maybe directly convert everything into a group state
        self.data_loader = None
        # fine tuning hyperparameters
        self.num_epochs = num_epochs
        self.batch_s = batch_size
        self.weight_decay = weight_decay
        self.lr_rate = lr_rate
        self.lr_decay_epoch = lr_decay_epoch
        self.lr_decay_gamma = lr_decay_gamma

        # maybe reduce this by only considereing best action and duplicating for other trajectories
        self.action_buffer = ActionBuffer()
        super().__init__(policy_net)
        self.decoder_copy = copy.deepcopy(self.model.node_prob_calculator)
        # prepare some wandb stuff for model tracking
        wandb.define_metric("adaptlr/epoch")
        wandb.define_metric("adaptlr/*", step_metric="adaptlr/epoch")

    def reset(self, state):
        # the magic happens here

        # encode nodes
        super().reset(state)
        self.encoded_nodes_full = self.model.encoded_nodes

        # first run greedy decoding and save results
        self.action_buffer.reset()
        greedy_lengths = self.solve_greedy(state)

        # initalize dataloader based on data
        self.data_loader = LocalDataLoader(
            state.data,
            self.encoded_nodes_full,
            greedy_lengths,
            self.batch_s,
            shuffle=True,
        )

        # run fine tuning (can be independent of group size but then training tours are harder to use)
        self.run_adaptlr(state.group_s, state, greedy_lengths)

        # restore initial decoder weights (maybe speed up somehow?)
        self.model.node_prob_calculator = copy.deepcopy(self.decoder_copy)

        # save all actions and output them one by one in get action

    def solve_greedy(self, state, soft_reset=False):
        self.model.node_prob_calculator.eval()
        if soft_reset:
            self.model.soft_reset(self.encoded_nodes_full)
        global_env = GroupEnvironment(state.data, state.tsp_size)
        global_env.reset(state.group_s)

        action_seq = []
        with torch.no_grad():
            done = global_env.is_done
            while not done:
                action_probs = self.get_action_probabilities(state)
                # shape = (batch, group, TSP_SIZE)
                action = action_probs.argmax(dim=2)
                action_seq.append(action)
                # shape = (batch, group)
                state, reward, done = global_env.step(action)
            max_reward, _ = reward.max(dim=1)
        # update action buffer (check if found tours are better)
        self.action_buffer.update(action_seq, -max_reward)
        return -max_reward

    def run_adaptlr(self, group_s, state, greedy_lengths):
        # at best take a dataloader based on the test dataset with the correct batchsize
        # otherwise init custom dataloader from a batch of data with the test batch size
        # (is strongly dependent on the test batch size and thus more complicated)

        # enable training on decoder weights --> check if really necessary (should be enabled by default)
        # self.model.unfreeze_decoder()

        self.setup_backprop()

        # make sure all output tensors have enabled grad for backprop
        with torch.enable_grad():
            for epoch in range(self.num_epochs):
                # fine tuning should also return the best action sequence encountered during epoch
                avg_tour_len, actor_loss_result = self.fine_tune_one_epoch(group_s)

                # potentially run greedy decoding after each fine tuning epoch for validation (and also use this )
                # compare with greedy len (for tuning on validation set)
                test_lengths = self.solve_greedy(state, soft_reset=True)
                test_avg_length = float(test_lengths.mean())
                improvement = (
                    float((self.action_buffer.best_lengths / greedy_lengths).mean()) - 1
                )
                # log metrics with wandb only after one epoch to prevent too much logging
                # log training behavior for each chunk of the data and also final performance
                wandb.log(
                    {
                        "adaptlr/epoch": epoch,
                        "adaptlr/train_loss": actor_loss_result,
                        "adaptlr/train_avg_length": avg_tour_len,
                        "adaptlr/test_avg_length": test_avg_length,
                        "adaptlr/test_improvement": improvement,
                    },
                    commit=True,
                )

    def fine_tune_one_epoch(self, group_s):
        self.model.node_prob_calculator.train()
        distance_AM = AverageMeter()
        actor_loss_AM = AverageMeter()
        episode = 0
        for batch_idx, (batch, encoded_nodes, lengths, indices) in enumerate(
            self.data_loader
        ):
            # should also return indices to restore order of the tsps for tracking best tour during training per instance
            # should also load
            # calculate node embeddings once --> can be made more efficient by calculating this for an independent batch size
            # node embeddings should only be calculated in the first epoch and then saved

            # implement policy gradient training algorithm

            batch_s = batch.size(0)
            episode = episode + batch_s
            tsp_size = batch.size(1)

            # Actor Group Move
            ###############################################
            local_env = GroupEnvironment(batch, tsp_size)
            group_state, reward, done = local_env.reset(group_size=group_s)
            # self.model.reset(group_state)
            self.model.soft_reset(encoded_nodes)
            # First Move is given
            # first_action = LongTensor(np.arange(group_s))[None, :].expand(
            #     batch_s, group_s
            # )
            # group_state, reward, done = local_env.step(first_action)

            group_prob_list = Tensor(np.zeros((batch_s, group_s, 0)))
            while not done:
                # actor_group.update(group_state)
                action_probs = self.model.get_action_probabilities(group_state)
                # shape = (batch, group, TSP_SIZE)
                action = (
                    action_probs.reshape(batch_s * group_s, -1)
                    .multinomial(1)
                    .squeeze(dim=1)
                    .reshape(batch_s, group_s)
                )
                # shape = (batch, group)
                group_state, reward, done = local_env.step(action)

                batch_idx_mat = torch.arange(batch_s)[:, None].expand(batch_s, group_s)
                group_idx_mat = torch.arange(group_s)[None, :].expand(batch_s, group_s)
                chosen_action_prob = action_probs[
                    batch_idx_mat, group_idx_mat, action
                ].reshape(batch_s, group_s)
                # shape = (batch, group)
                group_prob_list = torch.cat(
                    (group_prob_list, chosen_action_prob[:, :, None]), dim=2
                )
            max_reward, group_loss = self.learn(reward, group_prob_list)
            # if better exchange stored actions

            distance_AM.push(-max_reward)  # reward was given as negative dist
            actor_loss_AM.push(group_loss.detach().reshape(-1))
        actor_loss_result = actor_loss_AM.result()
        avg_tour_len = distance_AM.result()
        self.lr_stepper.step()
        return avg_tour_len, actor_loss_result

    def setup_backprop(self):
        # optimize only subset of parameters
        self.optimizer = optim.Adam(
            self.model.node_prob_calculator.parameters(),  # restrict parameters
            lr=self.lr_rate,
            weight_decay=self.weight_decay,
        )
        self.lr_stepper = lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.lr_decay_epoch,
            gamma=self.lr_decay_gamma,
        )

    def get_action(self, state):
        # return stored action sequence one by one (at best make this somehow depend on state to double check)
        action = self.action_buffer.get_action()
        action_info = None
        return action, action_info


class LocalDataset(Dataset):
    # return encoded nodes
    # greedy lengths
    # node coords
    # tsp index in batch
    def __init__(self, data, encoded_nodes, greedy_lengths) -> None:
        super().__init__()
        self.data = data
        self.encoded_nodes = encoded_nodes
        # self.encoded_nodes.requires_grad = True
        self.lengths = greedy_lengths

    def __getitem__(self, index):
        return (
            self.data[index][None],
            self.encoded_nodes[index][None],
            self.lengths[index][None],
            index,
        )

    def __len__(self):
        return int(self.data.size(0))


class LocalDataLoader(DataLoader):
    def __init__(self, data, encoded_nodes, greedy_lengths, batch_size, shuffle=False):
        super().__init__(
            dataset=LocalDataset(data, encoded_nodes, greedy_lengths),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=local_collate_fn,
            num_workers=0,
        )


def local_collate_fn(batch):
    # stack everything
    data = torch.cat([tup[0] for tup in batch])
    encoded_nodes = torch.cat([tup[1] for tup in batch])
    lengths = torch.cat([tup[2] for tup in batch])
    indices = [tup[3] for tup in batch]
    return data, encoded_nodes, lengths, indices


# def local_collate_fn(batch):
#     tmp_arr = np.array(batch, dtype=object)
#     # stack everything
#     data = torch.cat(tmp_arr[:, 0])
#     encoded_nodes = torch.cat(tmp_arr[:, 1])
#     lengths = torch.cat(tmp_arr[:, 2])
#     indices = tmp_arr[:, -1].astype(int)
#     return data, encoded_nodes, lengths, indices
