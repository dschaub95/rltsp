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


class TreeNode:
    def __init__(self, state, parent, prior_p, q_init=5, orig_prob=1.0, epsilon=0.5):
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
        # exploration constant
        self.epsilon = epsilon
        # only add for debugging
        self.orig_prob = orig_prob
        # add depth to a node for debugging
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    def expand(self, actions, priors, states=None):
        # fully expands a leaf considering all possible actions in this state
        # modify probabilitiy vector to favor exploration here
        probs_0 = priors.mean(dim=1).mean(dim=0).detach().cpu().numpy()
        # probs = priors.max(dim=1)[0].mean(dim=0).detach().cpu().numpy()
        # probs = priors.max(dim=1)[0].max(dim=0)[0].detach().cpu().numpy()
        # add exploration noise or rather smooth the probabilities
        if self.depth >= 0:
            # use softmax approach
            probs = 1.0 * probs_0 + self.epsilon
            probs = np.exp(probs) / np.sum(np.exp(probs))
            # use dirichlet approach as in alpha zero paper
            # probs = (1 - self.epsilon) * probs_0 + self.epsilon * np.random.dirichlet([1 for i in range(probs_0.shape[0])])
            pass
        for i, prob in enumerate(probs):
            action = actions[:, :, i]
            # state = states[i]
            if action not in self._children:
                self._children[action] = TreeNode(
                    None, self, prob, self.q_init, orig_prob=probs_0[i]
                )

    def select(self, c_puct):
        # select the best child
        rollout_values = [node._Q for node in self._children.values()]
        # print(rollout_values)
        mean_Q = np.mean(rollout_values)
        best_child = max(
            self._children.items(),
            key=lambda item: item[1].get_value(c_puct, self.max_Q, self.min_Q, mean_Q),
        )
        return best_child

    def add_visits(self, visits):
        if self._parent:
            self._parent.add_visits(visits)
        self._n_visits += visits

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
        # if the leaf was selected for the first time always use its value to overwrite any bad init value
        if self._n_visits == 1:
            self._Q = leaf_value
            self.max_Q = leaf_value
            self.min_Q = leaf_value
        else:
            self._Q = leaf_value if leaf_value > self._Q else self._Q

            self.max_Q = leaf_value if leaf_value > self.max_Q else self.max_Q
            self.min_Q = leaf_value if leaf_value < self.min_Q else self.min_Q

    def update_recursive(self, leaf_value):
        # max over trajectories and mean over batch for now
        if self._parent:
            self._parent.update_recursive(leaf_value)
        leaf_value = float(leaf_value.max(dim=-1)[0].max(dim=-1)[0])
        self.update(leaf_value)

    def get_value(self, c_puct, max_value, min_value, mean_value):
        # compute value for selection, higher is better
        # self._u = (c_puct * self._P * math.sqrt(self._parent._n_visits + 1) / (1 + self._n_visits))
        # alpha zero variant of puct
        self._u = (
            c_puct * self._P * math.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        )
        if max_value - min_value == 0:
            # q = self._Q
            # assign zero if the parent node (and potentially sub nodes) have been explored,
            # but giving a return not worse or better than the leaf calculated after first selection
            q = 0
        else:
            # q = -(self._Q - mean_value) / (max_value - min_value)
            # q = np.clip((self._Q - mean_value) / (max_value - min_value), 0, 1)
            q = (self._Q - mean_value) / (max_value - min_value)
            # q = 1 - (max_value - self._Q) / (max_value - min_value) # in [0,1]
        # print(q, self._Q, mean_value)
        value = q + self._u
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
    def __init__(
        self, net, c_puct=5, n_playout=400, n_parallel=2, virtual_loss=20, q_init=-5
    ):
        self._net = net
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.n_parallel = n_parallel
        self.virtual_loss = virtual_loss
        self.q_init = None
        self._root = None

    def initialize_search(self, env, group_size=1):
        self.env = env
        self.step = 0
        self.exploration_rating = 0
        state = self.env.initial_state(group_size=group_size)
        self._root = TreeNode(state, None, 1.0, self.q_init)
        return state

    def prepare_net(self, n_parallel):
        # prepare network once for parallelized leaf prediction
        num_nodes = self._net.encoded_nodes.size(-2)
        embed_dim = self._net.encoded_nodes.size(-1)
        encoded_nodes = (
            self._net.encoded_nodes[None, :, :, :]
            .expand(n_parallel, -1, -1, -1)
            .reshape(-1, num_nodes, embed_dim)
        )
        self._net.soft_reset(encoded_nodes)

    def select_leaf(self):
        current = self._root
        # copy root state and modify during selection to yield leaf state
        leaf_state = self._root.state.copy()
        # print(self.cur_playout)
        while True:
            if current.is_leaf():
                break
            act, current = current.select(self._c_puct)
            leaf_state = self.env.next_state(leaf_state, act)
            # print(act)
            # print(current.state.selected_node_list)
            # print(leaf_state.selected_node_list)
        current.state = leaf_state
        return current

    def _playout(self, num_parallel):
        leaves = []
        failsafe = 0
        # select a number of parallel leaves (each can have different number of taken action / depth in the tree)
        while len(leaves) < num_parallel and failsafe < num_parallel:
            failsafe += 1
            leaf = self.select_leaf()
            # add probability of the current leave to exploration rating
            self.exploration_rating += leaf.orig_prob
            if self.env.is_done_state(leaf.state):
                leaf_value = self.env.get_return(leaf.state)
                # increase the number of visits by one for all nodes belonging to this path
                # and update the value along the way
                leaf.update_recursive(leaf_value)
            else:
                # leaf.add_virtual_loss(self.virtual_loss)
                leaves.append(leaf)
            leaf.add_visits(visits=1)
            # # select the root node only once
            # if leaf is self._root:
            #     break
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
            # priors = self._eval([leaf.state for leaf in leaves])
            # Calc priors and values together
            values, priors = self.evaluate_leaves(leaves)
            # set the qinit and value after the first evaluation of the root node
            # based on the value following the rollout policy
            # min and max values are important for the selection process of child nodes
            if self.step == 0 and self.cur_playout == 0:
                root_value = float(values[0].max(dim=-1)[0].max(dim=-1)[0])
                self.q_init = 1.1 * root_value
                self._root.q_init = self.q_init
                self._root._Q = root_value
                self._root.min_Q = root_value
                self._root.max_Q = root_value
                pass
            for idx, (leaf, ps, value) in enumerate(zip(leaves, priors, values)):
                ### update_value
                # in 1d case convert tensor to float
                leaf.update_recursive(value)
                ### expand node considering all possible actions
                available_actions = leaf.state.available_actions
                num_ava_actions = available_actions.size(-1)
                # gather the probabilities for all the available actions
                # shape (batch_s, group_s, num_ava_actions)
                prior = torch.gather(ps, dim=-1, index=available_actions)
                prior, indices = torch.sort(prior, dim=-1, descending=True)
                available_actions = torch.gather(
                    available_actions, dim=-1, index=indices
                )
                # compute all possible states based on the available actions
                # next_states = [self.env.next_state(leaf.state, available_actions[:,:,i]) for i in range(num_ava_actions)]
                # expand all the leaves (each leave will be fully expanded)
                leaf.expand(available_actions, prior)  # , next_states)
                # delete state
                if leaf is not self._root:
                    del leaf.state
                    leaf.state = None

    def evaluate_leaves(self, leaves):
        # conversion to multi state
        multi_state = MultiGroupState([leaf.state.copy() for leaf in leaves])
        # potentially prepare net
        self.prepare_net(len(leaves))
        values, priors = self._value_func(multi_state)
        # restore net encoded nodes
        self._net.encoded_nodes = self._net.encoded_nodes[
            0 : multi_state.single_batch_s, :, :
        ]
        return values, priors
        # return self.value_func(leaves)
        # result = []
        # for leaf in leaves:
        #     result.append(self.beam_search(leaf.state))
        #     print(result[-1])
        #     time.txt.sleep(10)
        # return result
        # result = []
        # for leaf in leaves:
        #     result.append(self.random_rollout(leaf.state))
        # return result

    def random_rollout(self, state):
        # not used when we have given policy since policy can approximate the value by just taking actions
        while not self.env.is_done_state(state):
            # select any available action
            random_action = 0
            state = self.env.next_state(state, random_action)

        return self.env.get_return(state)

    def beam_search(self, state, k=5):
        sequences = [[state, 0.0]]
        while True:
            all_candidates = []
            # 遍历sequences中的每个元素
            for idx, sequence in enumerate(sequences):
                state, score = sequences[idx]
                priors = self._eval([state])[0]
                priors = priors[list(state["ava_action"])]

                for p, action in zip(priors, state["ava_action"]):
                    all_candidates.append(
                        [self.env.next_state(state, action), score - math.log(p + 1e-8)]
                    )

            sequences = []
            ordered = sorted(all_candidates, key=lambda tup: tup[1])
            for i in range(min(k, len(ordered))):
                sequences.append(ordered[i])

            if self.env.is_done_state(sequences[0][0]):
                return self.env.get_return(sequences[0][0])

    def _value_func(self, multi_state: MultiGroupState):
        # run the simulation
        # max_eval_count = np.min(multi_state.)
        for step in range(multi_state.n_actions - np.min(multi_state.selected_count)):
            action_probs = self._net.get_action_probabilities(multi_state)
            if step == 0:
                first_prob_tensor = action_probs
            action = action_probs.argmax(dim=2)
            self.env.next_state(multi_state, action)
        # collect simulation results
        states = multi_state.split()
        values = [self.env.get_return(state) for state in states]
        # split probability tensor into list for each leaf
        orig_size = multi_state.single_batch_s
        num_states = multi_state.num_single_states
        priors = [
            first_prob_tensor[i * orig_size : (i + 1) * orig_size, :, :]
            for i in range(num_states)
        ]
        return values, priors

    # def value_func(self, leaves):
    #     # calculate value for each leave by following the learned policy till the terminal state
    #     # leaved can have different number of selected nodes
    #     states = [leaf.state.copy() for leaf in leaves]
    #     # max_eval_count = 0
    #     # for leaf in leaves:
    #     #     max_eval_count = max(max_eval_count, leaf.state.n_actions - leaf.state.selected_count)
    #     max_eval_count = np.max([state.n_actions - state.selected_count for state in states])
    #     # combine all states into one
    #     for step in range(max_eval_count):
    #         action_probs = self._eval(states)
    #         if step == 0:
    #             priors = action_probs
    #         for idx, action_prob in enumerate(action_probs):
    #             action = action_prob.argmax(dim=2)
    #             # if self.step < 2 and step < 2:
    #             #     print("action probs", action_probs)
    #             #     print("action", action)
    #             states[idx] = self.env.next_state(states[idx], action)
    #     values = [self.env.get_return(state) for state in states]
    #     return values, priors

    # def _eval(self, states):
    #     # should be batched
    #     # write custom dataloader which splits combined data into batches
    #     # combine states
    #     multi_state = MultiGroupState(states)
    #     self.prepare_net(multi_state.num_single_states)
    #     # use initially prepared NN for prediction
    #     # use encoded nodes as input for forward pass
    #     # print([state.selected_node_list for state in states])
    #     prob_tensor = self._net.get_action_probabilities(multi_state)
    #     # prob_tensor.shape = (num_states*singl_batch_s, group_s, num_nodes)
    #     orig_size = multi_state.single_batch_s
    #     num_states = multi_state.num_single_states
    #     priors = [prob_tensor[i*orig_size:(i+1)*orig_size,:,:] for i in range(num_states)]
    #     # priors = [self._net.get_action_probabilities(state) for state in states]
    #     return priors

    def get_move_values(self):
        # add number of current simulations to n_playout to handle sub tree roots which have already been visited
        current_simulations = self._root._n_visits
        self.cur_playout = 0
        # while self._root._n_visits < self._n_playout + current_simulations:
        # print(f"Starting playout number {self.cur_playout}")
        while self.cur_playout < self._n_playout:
            self._playout(self.n_parallel)
            self.cur_playout += 1
        act_values_states = [
            (act, node._Q, node.state, node._P)
            for act, node in self._root._children.items()
        ]
        return zip(*act_values_states)

    def update_with_move(self, last_move):
        self.step += 1
        # delete all nodes that are not used anymore and keep the ones which are used
        last_state = self.env.next_state(self._root.state, last_move)
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root.state = last_state
            # transition root state based on action and save in new state
            self._root._parent = None
        else:
            self._root = TreeNode(last_state, None, 1.0, self.q_init)


class MCTSAgent(BaseAgent):
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
        self.mcts = MCTS(
            policy_net, c_puct, n_playout, n_parallel, virtual_loss, q_init
        )

    def reset(self, state: GroupState):
        # set different q_init for different tsp size
        # if state.tsp_size == 20:
        #     self.mcts.q_init = -5
        # elif state.tsp_size == 50:
        #     self.mcts.q_init = -7
        # elif state.tsp_size == 100:
        #     self.mcts.q_init = -10
        # init new environment model based on state
        # handle data batch sequentially for now
        env = GroupEnvironment(state.data, state.tsp_size)
        # initialize the mcts with the seperate environment model and return the starting state
        state = self.mcts.initialize_search(env, group_size=state.group_s)
        # reset the network and encode nodes once for this state
        super().reset(state)

    def get_action(self, state):
        # print(self.mcts._net.encoded_nodes)
        acts, values, states, priors = self.mcts.get_move_values()
        # print(values)
        # print(acts)
        # print(values)
        # print(states)

        # select best action based on values
        idx = np.argmax(values)
        # print(values)
        # print(priors)
        # update mcts tree
        self.mcts.update_with_move(acts[idx])
        # print(values)
        # if self.mcts.step >= 19:
        #     print(self.mcts.exploration_rating)
        return acts[idx]


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
        # set different q_init for different tsp size
        if state.tsp_size == 20:
            self.q_init = -5
        elif state.tsp_size == 50:
            self.q_init = -7
        elif state.tsp_size == 100:
            self.q_init = -10
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
