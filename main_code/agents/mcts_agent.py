# -*- coding: utf-8 -*-
import math
import random
import numpy as np
import torch

from main_code.agents.base_agent import BaseAgent
from main_code.environment.environment_new import GroupEnvironment, GroupState

class TreeNode:
    def __init__(self, state, parent, prior_p, q_init=5):
        self.state = state
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        # leaf value or expected leaf return
        self._Q = q_init
        self._u = 0
        self._P = prior_p
        self.q_init = q_init
        self.max_Q = q_init
        self.min_Q = q_init
        self.n_vlosses = 0
        # add depth to a node for debugging
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    def expand(self, actions, priors, states):
        # fully expands a leaf considering all possible actions in this state
        # convert tensors to suitable formats
        # actions is permuted before
        # priors
        priors = priors.cpu().numpy().squeeze((0, 1))
        for (action, prob, state) in zip(actions, priors, states):
            # permute dimensions again (only useful for batches)
            action = action.permute(1, 0)
            if action not in self._children:
                self._children[action] = TreeNode(state.copy(), self, prob, self.q_init)

    def select(self, c_puct):
        # select the best child
        mean_Q = np.mean([node._Q for node in self._children.values()])
        best_child = max(self._children.items(),
                         key=lambda item: item[1].get_value(c_puct, self.max_Q, self.min_Q, mean_Q))
        return best_child

    def add_virtual_loss(self, virtual_loss):
        if self._parent:
            self._parent.add_virtual_loss(virtual_loss)
        self.n_vlosses += 1
        self._n_visits += virtual_loss

    def revert_virtual_loss(self, virtual_loss):
        if self._parent:
            self._parent.add_virtual_loss(virtual_loss)
        self.n_vlosses -= 1
        self._n_visits -= virtual_loss

    def update(self, leaf_value):
        self._n_visits += 1

        self._Q = leaf_value if leaf_value > self._Q else self._Q

        self.max_Q = leaf_value if leaf_value > self.max_Q else self.max_Q
        self.min_Q = leaf_value if leaf_value < self.min_Q else self.min_Q

    def update_recursive(self, leaf_value):
        leaf_value = float(leaf_value)
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct, max_value, min_value, mean_value):
        # compute value for selection, higher is better
        self._u = (c_puct * self._P * math.sqrt(self._parent._n_visits + 1) / (1 + self._n_visits))
        if max_value - min_value == 0:
            value = -self._Q + self._u
        else:
            value = -(self._Q - mean_value) / (max_value - min_value) + self._u
        # print(value)
        return value
    
    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None

    def __repr__(self) -> str:
        return str(self.__dict__)

    def print_sub_tree(self):
        # print actions layerwise for the entire tree
        nodes = [int(item[0]) for item in self._children.items()]
        print(nodes)


class MCTS:
    def __init__(self, net, c_puct=5, n_playout=400, n_parallel=2, virtual_loss=20, q_init=-5):
        self._net = net
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.n_parallel = n_parallel
        self.virtual_loss = virtual_loss
        self.q_init = q_init
        self._root = None
        self.step = 0

    def initialize_search(self, env, group_size=1):
        self.env = env
        state = self.env.initial_state(group_size=group_size)
        self._root = TreeNode(state, None, 1.0, self.q_init)
        return state

    def select_leaf(self):
        current = self._root
        while True:
            if current.is_leaf():
                break
            _, current = current.select(self._c_puct)

        return current

    def _playout(self, num_parallel):
        leaves = []
        failsafe = 0
        # select a number of parallel leaves (each can have different number of taken action / depth in the tree)
        while len(leaves) < num_parallel and failsafe < num_parallel:
            failsafe += 1
            leaf = self.select_leaf()
            if self.env.is_done_state(leaf.state):
                leaf_value = self.env.get_return(leaf.state)
                # increase the number of visits by one for all nodes belonging to this path
                # and update the value along the way
                leaf.update_recursive(leaf_value)
            else:
                # leaf.add_virtual_loss(self.virtual_loss)
                leaves.append(leaf)

        if leaves:
            # if self.step < 2:
            #     print("step:", self.step)
            #     print("play out number:", self.cur_playout)
            #     print("leaves states selected nodes:", [leaf.state.selected_node_list for leaf in leaves])
            #     # print("leaves states data:")
            #     print("leaves states ninfmasks:", [leaf.state.ninf_mask for leaf in leaves])
            #     # print("Leaves depths",[leaf.depth for leaf in leaves])
            # revert_virtual_loss
            # for leaf in leaves:
            #     leaf.revert_virtual_loss(self.virtual_loss)
            # Calc priors, can be parallelized
            priors = self._eval([leaf.state for leaf in leaves])
            # priors = np.ones((len(leaves), leaves[0].state['graph'].ver_num))
            # Cla values
            values = self.evaluate_leaves(leaves)
            for idx, (leaf, ps, value) in enumerate(zip(leaves, priors, values)):
                ### update_value
                # in 1d case convert tensor to float
                leaf.update_recursive(value)
                ### expand node considering all possible actions
                available_actions = leaf.state.available_actions
                # gather the probabilities for all the available actions
                prior = torch.gather(ps, -1, available_actions)
                permuted_available_actions = available_actions.permute(2, 1, 0)
                # compute all possible states based on the available actions
                next_states = [self.env.next_state(leaf.state.copy(), act.permute(1, 0)) for act in permuted_available_actions]
                # expand all the leaves (each leave will be fully expanded)
                leaf.expand(permuted_available_actions, prior, next_states)

    def evaluate_leaves(self, leaves):
        # result = []
        # for leaf in leaves:
        #     result.append(self.beam_search(leaf.state))
        #     print(result[-1])
        #     time.txt.sleep(10)
        # return result

        return self.value_func(leaves)

        # result = []
        # for leaf in leaves:
        #     result.append(self.random_rollout(leaf.state))
        # return result

    def random_rollout(self, state):
        # not used when we have given policy since policy can approximate the value by just taking actions
        while not self.env.is_done_state(state):
            # select any available action
            state = self.env.next_state(state, random.choice(list(state['ava_action'])))

        return self.env.get_return(state)

    def beam_search(self, state, k=5):
        sequences = [[state, 0.0]]
        while True:
            all_candidates = []
            # 遍历sequences中的每个元素
            for idx, sequence in enumerate(sequences):
                state, score = sequences[idx]
                priors = self._eval([state])[0]
                priors = priors[list(state['ava_action'])]

                for p, action in zip(priors, state['ava_action']):
                    all_candidates.append([self.env.next_state(state, action), score - math.log(p + 1e-8)])

            sequences = []
            ordered = sorted(all_candidates, key=lambda tup: tup[1])
            for i in range(min(k, len(ordered))):
                sequences.append(ordered[i])

            if self.env.is_done_state(sequences[0][0]):
                return self.env.get_return(sequences[0][0])

    def value_func(self, leaves):
        # calculate value for each leave by following the learned policy till the terminal state
        # leaved can have different number of selected nodes
        states = [leaf.state.copy() for leaf in leaves]
        max_eval_count = 0
        for leaf in leaves:
            max_eval_count = max(max_eval_count, leaf.state.n_actions - leaf.state.selected_count)
        for step in range(max_eval_count):
            action_probs = self._eval([state for state in states])
            for idx, action_prob in enumerate(action_probs):
                action = action_prob.argmax(dim=2)
                # if self.step < 2 and step < 2:
                #     print("action probs", action_probs)
                #     print("action", action)
                states[idx] = self.env.next_state(states[idx], action)
        values = [self.env.get_return(state) for state in states]
        # if self.step < 2:
        #     print("leaves values:", values)
        return values

    def _eval(self, states):
        # should be batched
        # print([state.selected_node_list for state in states])
        priors = [self._net.get_action_probabilities(state) for state in states]
        return priors

    def get_move_values(self):
        # add number of current simulations to n_playout to handle sub tree roots which have already been visited
        current_simulations = self._root._n_visits
        self.cur_playout = 0
        # while self._root._n_visits < self._n_playout + current_simulations:
            # print(f"Starting playout number {self.cur_playout}")
        while self.cur_playout < self._n_playout:
            self._playout(self.n_parallel)
            self.cur_playout += 1
        act_values_states = [(act, node._Q, node.state) for act, node in self._root._children.items()]
        return zip(*act_values_states)

    def update_with_move(self, last_move, last_state):
        self.step += 1
        # delete all nodes that are not used anymore and keep the ones which are used
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(last_state, None, 1.0, self.q_init)

    def print_mct(self):
        pass

class MCTSAgent(BaseAgent):
    def __init__(self, policy_net, c_puct=1.3, n_playout=10, n_parallel=1, virtual_loss=0, q_init=-5) -> None:
        super().__init__(policy_net)
        self.mcts = MCTS(policy_net, c_puct, n_playout, n_parallel, virtual_loss, q_init)
    
    def reset(self, state: GroupState):
        # init new environment model based on state
        # handle data batch sequentially for now
        env = GroupEnvironment(state.data, state.tsp_size)
        # initialize the mcts with the seperate environment model and return the starting state
        state = self.mcts.initialize_search(env)
        # reset the network and encode nodes once for this state
        super().reset(state)
    
    def get_action(self, state):
        # print(self.mcts._net.encoded_nodes)
        acts, values, states = self.mcts.get_move_values()
        # print(values)
        # print(acts)
        # print(values)
        # print(states)

        # select best action based on values
        idx = np.argmax(values)
        # update mcts tree
        self.mcts.update_with_move(acts[idx], states[idx])
        # print(values)
        return acts[idx]


class MCTSBatchAgent(BaseAgent):
    def __init__(self, policy_net, c_puct=1.3, n_playout=10, n_parallel=1, virtual_loss=0, q_init=-5) -> None:
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
        self.encoded_nodes_list = [encoded_nodes.unsqueeze(0) for encoded_nodes in self.model.encoded_nodes]
        states = state.split_along_batch_dim()
        env_list = [GroupEnvironment(state.data, state.tsp_size) for state in states]
        self.mcts_list = [MCTS(self.policy_net, self.c_puct, self.n_playout, self.n_parallel, self.virtual_loss, self.q_init) for env in env_list]
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
            acts, values, states = mcts.get_move_values()
            # print(values)
            idx = np.argmax(values)
            # update mcts tree
            mcts.update_with_move(acts[idx], states[idx])
            batch_action.append(acts[idx])
        batch_action = torch.cat(batch_action, dim=0)
        return batch_action