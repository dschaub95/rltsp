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
        action = self.best_actions[:, :, self.counter]
        self.counter += 1
        return action

    def update(self, final_state, lengths, indices=None):
        # update the memory if an action sequence for any sample is better
        # first check lengths
        action_tensor = final_state.selected_node_list[:, :, 1::]
        if self.best_actions is None:
            self.best_actions = action_tensor
            self.best_lengths = lengths
        else:
            if indices is None:
                # directly compare lengths and current best lengths
                mask = lengths < self.best_lengths
                self.best_actions[mask] = action_tensor[mask]
                self.best_lengths[mask] = lengths[mask]
            else:
                # compare lengths by extracting lengths based on indices
                # desired version
                # mask = lengths < self.best_lengths[indices]
                # # self.best_actions[indices][mask] = action_tensor[mask]
                # tmp_actions = self.best_actions[indices]
                # tmp_actions[mask] = action_tensor[mask]
                # self.best_actions[indices] = tmp_actions
                # # self.best_lengths[indices][mask] = lengths[mask]
                # tmp_lengths = self.best_lengths[indices]
                # tmp_lengths[mask] = lengths[mask]
                # self.best_lengths[indices] = tmp_lengths
                mask = lengths < self.best_lengths[indices]
                self.best_actions[indices[mask]] = action_tensor[mask]
                self.best_lengths[indices[mask]] = lengths[mask]

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
        state_space_size=64,
        lr_rate=1e-4,
        weight_decay=1e-6,
        lr_decay_epoch=1.0,
        lr_decay_gamma=1.0,
        noise_factor=0.0,
    ) -> None:
        # extra hyperpramaters for learning as mcts
        # specifically should have parameter defining the batch size for learning and the state space size
        # and standard learning parameters
        # data loader should return embedded nodes and maybe directly convert everything into a group state
        self.train_loader = None
        # fine tuning hyperparameters
        self.num_epochs = num_epochs
        self.batch_s = batch_size
        self.state_space_s = state_space_size
        self.weight_decay = weight_decay
        self.lr_rate = lr_rate
        self.lr_decay_epoch = lr_decay_epoch
        self.lr_decay_gamma = lr_decay_gamma
        self.noise_factor = noise_factor

        # maybe reduce this by only considereing best action and duplicating for other trajectories
        self.action_buffer = ActionBuffer()
        super().__init__(policy_net)
        self.decoder_copy = copy.deepcopy(self.model.node_prob_calculator)
        # prepare some wandb stuff for model tracking
        wandb.define_metric("adaptlr/epoch")
        wandb.define_metric("adaptlr/*", step_metric="adaptlr/epoch")

    def reset(self, state):
        # the magic happens here
        self.tsp_size = state.tsp_size
        self.group_s = state.group_s
        # encode nodes once
        super().reset(state)
        self.encoded_nodes_all = self.model.encoded_nodes

        # run initial greedy decoding and save results in action buffer
        self.action_buffer.reset()
        greedy_lengths = self.solve_greedy(state.data, self.group_s)

        # run fine tuning (can be independent of group size but then training tours are harder to use)
        self.run_adaptlr(state, greedy_lengths, self.group_s)

    def restore_decoder(self):
        # restore initial decoder weights (maybe speed up somehow?)
        self.model.node_prob_calculator = copy.deepcopy(self.decoder_copy)

    def solve_greedy(self, data, group_s, encoded_nodes=None, indices=None):
        self.model.node_prob_calculator.eval()
        if encoded_nodes is not None:
            self.model.soft_reset(encoded_nodes)
        global_env = GroupEnvironment(data, self.tsp_size)
        state, reward, done = global_env.reset(group_s)

        with torch.no_grad():
            done = global_env.is_done
            while not done:
                action_probs = self.get_action_probabilities(state)
                # shape = (batch, group, TSP_SIZE)
                action = action_probs.argmax(dim=2)
                # shape = (batch, group)
                state, reward, done = global_env.step(action)
            max_reward, _ = reward.max(dim=1)
        # update action buffer (check if found tours are better)
        self.action_buffer.update(state, -max_reward, indices)
        return -max_reward

    def run_adaptlr(self, state, greedy_lengths, group_s):
        # initalize local dataloader based on data
        state_loader = LocalDataLoader(
            state.data,
            self.encoded_nodes_all,
            greedy_lengths,
            self.state_space_s,
            shuffle=False,
        )
        for space_batch, space_encoded_nodes, lengths, space_indices in state_loader:
            # set new seed for reproducibility
            torch.manual_seed(42)
            train_loader = LocalDataLoader(
                space_batch,
                space_encoded_nodes,
                lengths,
                self.batch_s,
                shuffle=True,
                indices=space_indices,
            )

            # self.add_noise_to_decoder(self.noise_factor)
            self.setup_backprop()

            # make sure all output tensors have enabled grad for backprop
            with torch.enable_grad():
                for epoch in range(self.num_epochs):
                    # run fine tuning and update action buffer if improved tour was found during training
                    avg_tour_len, actor_loss_result = self.fine_tune_one_epoch(
                        train_loader, group_s
                    )
                    # run greedy decoding after each fine tuning epoch
                    test_lengths = self.solve_greedy(
                        space_batch,
                        group_s,
                        encoded_nodes=space_encoded_nodes,
                        indices=space_indices,
                    )
                    test_avg_length = float(test_lengths.mean())
                    improvement = (
                        float((self.action_buffer.best_lengths / greedy_lengths).mean())
                        - 1
                    )
                    # log metrics with wandb only after one epoch to prevent too much logging
                    # log training behavior for each chunk of the data and also final performance
                    # aggregate all training metrics into table and upload after fine tuning
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
            self.restore_decoder()

    def fine_tune_one_epoch(self, train_loader, group_s):
        self.model.node_prob_calculator.train()
        distance_AM = AverageMeter()
        actor_loss_AM = AverageMeter()
        for batch_idx, (batch, encoded_nodes, lengths, indices) in enumerate(
            train_loader
        ):
            # should also return indices to restore order of the tsps for tracking best tour during training per instance
            # calculate node embeddings once --> can be made more efficient by calculating this for an independent batch size
            # node embeddings should only be calculated in the first epoch and then saved

            # implement policy gradient training algorithm

            batch_s = batch.size(0)

            # Actor Group Move
            ###############################################
            local_env = GroupEnvironment(batch, self.tsp_size)
            group_state, reward, done = local_env.reset(group_size=group_s)
            # self.model.reset(group_state)
            self.model.soft_reset(encoded_nodes)

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
            self.action_buffer.update(group_state, -max_reward, indices=indices)

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

    def add_noise_to_decoder(self, noise_factor=0.1):
        with torch.no_grad():
            for param in self.model.node_prob_calculator.parameters():
                param.add_(torch.randn(param.size(), device=device) * noise_factor)

    def get_action(self, state):
        # return stored action sequence one by one
        action = self.action_buffer.get_action()
        action_info = None
        return action, action_info


class LocalDataset(Dataset):
    # return encoded nodes
    # greedy lengths
    # node coords
    # tsp index in batch
    def __init__(
        self, data, encoded_nodes, greedy_lengths, batch_size, indices=None
    ) -> None:
        super().__init__()
        self.data = data
        self.data_size = int(self.data.size(0))
        self.encoded_nodes = encoded_nodes
        self.lengths = greedy_lengths
        self.indices = indices
        self.batch_s = batch_size

    def __getitem__(self, index):
        # if batch_s is bigger that data size
        index = index % self.data_size
        if self.indices is not None:
            true_index = self.indices[index]
        else:
            true_index = index
        return (
            self.data[index][None],
            self.encoded_nodes[index][None],
            self.lengths[index][None],
            true_index,
        )

    def __len__(self):
        return max(self.batch_s, self.data_size)


class LocalDataLoader(DataLoader):
    def __init__(
        self,
        data,
        encoded_nodes,
        greedy_lengths,
        batch_size,
        shuffle=False,
        indices=None,
    ):
        super().__init__(
            dataset=LocalDataset(
                data, encoded_nodes, greedy_lengths, batch_size, indices
            ),
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
    indices = LongTensor([tup[3] for tup in batch])
    return data, encoded_nodes, lengths, indices
