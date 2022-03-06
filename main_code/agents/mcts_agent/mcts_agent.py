# -*- coding: utf-8 -*-
import math
import random
import numpy as np
import torch
from main_code.utils.torch_objects import FloatTensor, device
from main_code.agents.base_agent import BaseAgent
from main_code.environment.environment_new import (
    GroupEnvironment,
    GroupState,
    MultiGroupState,
)
from main_code.agents.mcts_agent.mcts import MCTS


class MCTSAgent(BaseAgent):
    def __init__(
        self,
        policy_net,
        mcts_config,
    ) -> None:
        super().__init__(policy_net)
        self.mcts = MCTS(policy_net, **mcts_config)

    def reset(self, state: GroupState):
        # init new environment model based on state
        # handle data batch sequentially for now
        env = GroupEnvironment(state.data, state.tsp_size)
        # initialize the mcts with the seperate environment model and return the starting state
        state = self.mcts.initialize_search(env, group_size=state.group_s)
        # reset the network and encode nodes once for this state
        super().reset(state)

    def get_action(self, state):
        # print(self.mcts._net.encoded_nodes)
        acts, values, states, priors, n_visits = self.mcts.get_move_values()
        # select best action based on values
        idx = np.argmax(values)
        # update mcts tree
        self.mcts.update_with_move(acts[idx])
        # print(values)
        # if self.mcts.step >= 19:
        #     print(self.mcts.exploration_rating)
        action_info = {
            "orig_prob_action": priors[idx],
            "values": values,
            "priors": priors,
            "n_visits": n_visits,
        }
        return acts[idx], action_info


class MCTSBatchAgent(BaseAgent):
    def __init__(
        self,
        policy_net,
        c_puct=1.3,
        n_playout=10,
        n_parallel=1,
        virtual_loss=0,
        q_init=-5,
    ) -> None:
        super().__init__(policy_net)
        self.policy_net = policy_net
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.n_parallel = n_parallel
        self.virtual_loss = virtual_loss
        self.q_init = q_init

    def reset(self, state: GroupState):
        # make batch calculation for the encoded nodes
        self.model.reset(state)
        self.encoded_nodes_list = [
            encoded_nodes.unsqueeze(0) for encoded_nodes in self.model.encoded_nodes
        ]
        states = state.split_along_batch_dim()
        env_list = [GroupEnvironment(state.data, state.tsp_size) for state in states]
        self.mcts_list = [
            MCTS(
                self.policy_net,
                self.c_puct,
                self.n_playout,
                self.n_parallel,
                self.virtual_loss,
                self.q_init,
            )
            for env in env_list
        ]
        # init mcts
        for i, mcts in enumerate(self.mcts_list):
            mcts.initialize_search(env_list[i], group_size=state.group_s)

    def get_action(self, state):
        # iterate over mc tree for each sample in the batch
        batch_action = []
        for k, mcts in enumerate(self.mcts_list):
            # prepare the network
            mcts._net.soft_reset(self.encoded_nodes_list[k])
            # print(mcts._net.encoded_nodes)
            acts, values, states, priors = mcts.get_move_values()
            # print(values)
            idx = np.argmax(values)
            # update mcts tree
            mcts.update_with_move(acts[idx])
            batch_action.append(acts[idx])
        batch_action = torch.cat(batch_action, dim=0)
        return batch_action
