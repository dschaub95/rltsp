import torch
import numpy as np
from torch.functional import split

from main_code.utils.torch_objects import BoolTensor, Tensor, LongTensor


class GroupState:
    def __init__(self, group_size, data, tsp_size):
        # data.shape = (batch, group, 2)
        self.batch_s = data.size(0)
        self.group_s = group_size
        self.data = data
        self.tsp_size = tsp_size
        self.n_actions = tsp_size
        self.reset()

    def reset(self):
        # History
        ####################################
        self.selected_count = 0
        self.current_node = None
        # shape = (batch, group)
        self.selected_node_list = LongTensor(np.zeros((self.batch_s, self.group_s, 0)))
        # shape = (batch, group, selected_count)

        # Status
        ####################################
        self.ninf_mask = Tensor(np.zeros((self.batch_s, self.group_s, self.tsp_size)))
        # shape = (batch, group, tsp_size)
    
    def transition(self, selected_idx_mat):
        # selected_idx_mat.shape = (batch, group)

        # History update
        ####################################
        self.selected_count += 1
        self.current_node = selected_idx_mat
        self.selected_node_list = torch.cat((self.selected_node_list, selected_idx_mat[:, :, None]), dim=2)
        # Status
        ####################################
        batch_idx_mat = torch.arange(self.batch_s)[:, None].expand(self.batch_s, self.group_s)
        group_idx_mat = torch.arange(self.group_s)[None, :].expand(self.batch_s, self.group_s)
        self.ninf_mask[batch_idx_mat, group_idx_mat, selected_idx_mat] = -np.inf
    
    def compute_path_len(self):
        # ignore any additional nodes which might be added for batching reasons
        self.selected_node_list = self.selected_node_list[:, :, 0:self.tsp_size]
        gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_s, -1, self.tsp_size, 2)
        # shape = (batch, group, tsp_size, 2)
        seq_expanded = self.data[:, None, :, :].expand(self.batch_s, self.group_s, self.tsp_size, 2)
        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape = (batch, group, tsp_size, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # size = (batch, group, tsp_size)

        group_travel_distances = segment_lengths.sum(2)
        # size = (batch, group)
        return group_travel_distances
    
    def split_along_batch_dim(self):
        split_group_states = []
        for i in range(self.batch_s):
            group_state = GroupState(self.group_s, self.data[i].unsqueeze(0), self.tsp_size)
            group_state.selected_count = self.selected_count
            group_state.current_node = self.current_node[i].unsqueeze(0).detach().clone()
            group_state.selected_node_list = self.selected_node_list[i].unsqueeze(0).detach().clone()
            group_state.ninf_mask = self.ninf_mask[i].unsqueeze(0).detach().clone()
            split_group_states.append(group_state)
        return split_group_states

    def copy(self):
        state_copy = GroupState(self.group_s, self.data, self.tsp_size)
        state_copy.selected_count = self.selected_count
        state_copy.current_node = self.current_node.detach().clone()
        state_copy.selected_node_list = self.selected_node_list.detach().clone()
        state_copy.ninf_mask = self.ninf_mask.detach().clone()
        return state_copy

    @property
    def available_actions(self):
        # return all currently available actions
        full_node_list = LongTensor(np.arange(self.tsp_size))[None, None, :].expand(self.batch_s, self.group_s, self.tsp_size)
        mask = torch.ones_like(full_node_list).scatter_(2, self.selected_node_list, 0.)
        return full_node_list[mask.bool()].view(self.batch_s, self.group_s, self.tsp_size - self.selected_count)
        # shape (batch_s, group_s, tsp_size - selected_count)

class GroupEnvironment:
    def __init__(self, data, tsp_size):
        # seq.shape = (batch, tsp_size, 2)

        self.data = data
        self.batch_s = data.size(0)
        self.group_s = None
        self.group_state = None
        self.tsp_size = tsp_size
        self.done = False

    def reset(self, group_size):
        self.group_s = group_size
        self.group_state = GroupState(group_size=group_size, data=self.data, tsp_size=self.tsp_size)
        reward = None
        self.done = False

        # First Move is given by default
        first_action = LongTensor(np.arange(self.group_s))[None, :].expand(self.batch_s, self.group_s)
        self.group_state, reward, self.done = self.step(first_action)
        return self.group_state, reward, self.done

    def initial_state(self, group_size):
        group_state, reward, done = self.reset(group_size)
        return group_state

    def step(self, selected_idx_mat):
        # selected_idx_mat.shape = (batch, group)

        # move state
        self.group_state.transition(selected_idx_mat)

        # returning values
        self.done = (self.group_state.selected_count == self.tsp_size)
        if self.done:
            reward = -self._get_group_travel_distance()  # note the minus sign!
        else:
            reward = None
        return self.group_state, reward, self.done

    def _get_group_travel_distance(self):
        return self.group_state.compute_path_len()
    
    @property
    def isdone(self):
        return (self.group_state.selected_count == self.tsp_size)
    
    @staticmethod
    def is_done_state(group_state: GroupState):
        return (group_state.selected_count == group_state.tsp_size)
    
    @staticmethod
    def next_state(group_state: GroupState, action):
        group_state.transition(action)
        return group_state

    @staticmethod
    def get_return(group_state: GroupState):
        # assert group_state.selected_count == group_state.tsp_size
        return -group_state.compute_path_len()