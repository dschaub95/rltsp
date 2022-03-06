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
        self.selected_node_list = torch.cat(
            (self.selected_node_list, selected_idx_mat[:, :, None]), dim=2
        )
        # Status
        ####################################
        batch_idx_mat = torch.arange(self.batch_s)[:, None].expand(
            self.batch_s, self.group_s
        )
        group_idx_mat = torch.arange(self.group_s)[None, :].expand(
            self.batch_s, self.group_s
        )
        self.ninf_mask[batch_idx_mat, group_idx_mat, selected_idx_mat] = -np.inf

    def compute_path_len(self):
        # ignore any additional nodes which might be added for batching reasons
        self.selected_node_list = self.selected_node_list[:, :, 0 : self.tsp_size]
        gathering_index = self.selected_node_list.unsqueeze(3).expand(
            self.batch_s, -1, self.tsp_size, 2
        )
        # shape = (batch, group, tsp_size, 2)
        seq_expanded = self.data[:, None, :, :].expand(
            self.batch_s, self.group_s, self.tsp_size, 2
        )
        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape = (batch, group, tsp_size, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # size = (batch, group, tsp_size)

        group_travel_distances = segment_lengths.sum(2)
        # size = (batch, group)
        return group_travel_distances

    def split_along_batch_dim(self):
        split_group_states = []
        for i in range(self.batch_s):
            group_state = GroupState(
                self.group_s, self.data[i].unsqueeze(0), self.tsp_size
            )
            group_state.selected_count = self.selected_count
            group_state.current_node = (
                self.current_node[i].unsqueeze(0).detach().clone()
            )
            group_state.selected_node_list = (
                self.selected_node_list[i].unsqueeze(0).detach().clone()
            )
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
        full_node_list = LongTensor(np.arange(self.tsp_size))[None, None, :].expand(
            self.batch_s, self.group_s, self.tsp_size
        )
        mask = torch.ones_like(full_node_list).scatter_(2, self.selected_node_list, 0.0)
        return full_node_list[mask.bool()].view(
            self.batch_s, self.group_s, self.tsp_size - self.selected_count
        )
        # shape (batch_s, group_s, tsp_size - selected_count)

    # @property
    # def selected_count(self):


class MultiGroupState(GroupState):
    def __init__(self, state_list):
        # combine states
        data = torch.cat([state.data for state in state_list], dim=0)
        group_size = state_list[0].group_s
        tsp_size = state_list[0].tsp_size
        self.single_batch_s = state_list[0].batch_s
        self.num_single_states = len(state_list)
        # sets the correct batch size
        super().__init__(group_size, data, tsp_size)
        # set selected count as vector to allow addition of ints
        self.selected_count = np.array([state.selected_count for state in state_list])
        max_count = np.max(self.selected_count)
        # pad selected_node list tensors to the left
        self.selected_node_list = torch.cat(
            [
                self.pad3d_left(state.selected_node_list, max_count)
                for state in state_list
            ],
            dim=0,
        )
        self.ninf_mask = torch.cat([state.ninf_mask for state in state_list], dim=0)
        self.current_node = torch.cat(
            [state.current_node for state in state_list], dim=0
        )

    # overwrite available actions
    def pad3d_left(self, tensor, desired_size: int):
        orig_size = tensor.size(-1)
        tmp = tensor[:, :, 0][:, :, None].expand(-1, -1, desired_size).detach().clone()
        tmp[:, :, -orig_size::] = tensor
        return tmp

    # adding to selected count works for numpy array
    def transition(self, selected_idx_mat):
        return super().transition(selected_idx_mat)

    def compute_path_len(self):
        # reduce to single state case
        states = self.split()
        return torch.cat([state.compute_path_len() for state in states], dim=0)

    def split(self):
        # split multi group state into original single group states
        orig_batch_s = self.single_batch_s
        orig_data = self.data[0:orig_batch_s, :, :]
        # orig_data.shape = (batch_s, tsp_size, 2)
        states = []
        for k, selected_count in enumerate(self.selected_count):
            state = GroupState(self.group_s, data=orig_data, tsp_size=self.tsp_size)
            state.selected_count = np.clip(selected_count, 0, self.tsp_size)
            state.ninf_mask = self.ninf_mask[
                k * orig_batch_s : (k + 1) * orig_batch_s, :, :
            ]
            # recover the true selected node tensor
            raw_selected_node_list = self.selected_node_list[
                k * orig_batch_s : (k + 1) * orig_batch_s, :, :
            ]
            num_selected_nodes = raw_selected_node_list.size(-1)
            # recover single selected node list
            state.selected_node_list = torch.unique_consecutive(
                raw_selected_node_list, dim=-1
            )[:, :, 0 : self.tsp_size]
            # recover last selected or current node
            state.current_node = state.selected_node_list[:, :, -1]
            states.append(state)
        return states


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
        self.group_state = GroupState(
            group_size=group_size, data=self.data, tsp_size=self.tsp_size
        )
        reward = None
        self.done = False

        # First Move is given by default
        first_action = LongTensor(np.arange(self.group_s))[None, :].expand(
            self.batch_s, self.group_s
        )
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
        self.done = self.group_state.selected_count == self.tsp_size
        if self.is_done:
            reward = -self._get_group_travel_distance()  # note the minus sign!
        else:
            reward = None
        return self.group_state, reward, self.is_done

    def _get_group_travel_distance(self):
        return self.group_state.compute_path_len()

    @property
    def is_done(self):
        return self.group_state.selected_count == self.tsp_size

    @staticmethod
    def is_done_state(group_state: GroupState):
        return group_state.selected_count == group_state.tsp_size

    @staticmethod
    def next_state(group_state: GroupState, action):
        group_state.transition(action)
        return group_state

    @staticmethod
    def get_return(group_state: GroupState):
        # assert group_state.selected_count == group_state.tsp_size
        return -group_state.compute_path_len()
