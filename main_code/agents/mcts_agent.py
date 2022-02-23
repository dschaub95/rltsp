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
    def __init__(self, state, parent, prior_p, q_init=None, orig_prob=1.0):
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
        # only add for debugging
        self.orig_prob = orig_prob
        # add depth to a node for debugging
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    def _transform_policy(self, probs, epsilon):
        # use softmax approach
        # probs = 1.0 * probs_0 + self.epsilon
        # probs = np.exp(probs) / np.sum(np.exp(probs))
        # use dirichlet approach as in alpha zero paper
        # probs = (1 - self.epsilon) * probs_0 + self.epsilon * np.random.dirichlet([1 for i in range(probs_0.shape[0])])
        # use bayesian mixing with the uniform policy for some epsilon > 0
        uniform_probs = np.ones(probs.shape) * 1 / probs.shape[0]
        probs = (1 - epsilon) * probs + epsilon * uniform_probs
        return probs

    def expand(
        self, actions, priors, epsilon, aggregation_strategy, expansion_limit=None
    ):
        # fully expands a leaf considering all possible actions in this state
        # modify probabilitiy vector to favor exploration here
        if aggregation_strategy == 0:
            probs_0 = priors.mean(dim=1).mean(dim=0).detach().cpu().numpy()
        # probs_0 = priors.max(dim=1)[0].mean(dim=0).detach().cpu().numpy()
        # probs_0 = priors.max(dim=1)[0].max(dim=0)[0].detach().cpu().numpy()
        # add exploration noise or rather smooth the probabilities
        if self.depth >= 0:
            probs = self._transform_policy(probs_0, epsilon)
        # only expand kth most promising children
        if expansion_limit is None:
            k = probs.shape[0]
        else:
            k = np.clip(expansion_limit, 1, probs.shape[0])
        for i, prob in enumerate(probs[0:k]):
            action = actions[:, :, i]
            # state = states[i]
            if action not in self._children:
                self._children[action] = TreeNode(
                    None, self, prob, self.q_init, orig_prob=probs_0[i]
                )

    def select(
        self, c_puct, node_value_term, node_value_scale, prob_term, weight_fac=50
    ):
        # give all value to the value calculation --> selection function should not change, just the formula
        child_Qs = [node._Q for node in self._children.values()]
        mean_Q = np.mean(child_Qs)
        # child_Qs = [q for q in child_Qs if q is not None]
        # test epsilon greedy action selection
        # select the best child
        best_child = max(
            self._children.items(),
            key=lambda item: item[1].get_value(
                c_puct, mean_Q, node_value_term, node_value_scale, prob_term, weight_fac
            ),
        )
        return best_child

    def add_visits(self, visits):
        if self._parent:
            self._parent.add_visits(visits)
        self._n_visits += visits

    def update(self, leaf_value):
        self._n_visits += 1
        # if the leaf was selected for the first time or is terminal always use its value to overwrite any bad init value
        if self._n_visits <= 1:
            # always keep initial q value as base line for evaluation of the leaf nodes
            self.q_init = leaf_value
            self._Q = leaf_value
            self.max_Q = leaf_value
            self.min_Q = leaf_value
        else:
            self._Q = leaf_value if leaf_value > self._Q else self._Q

            self.max_Q = leaf_value if leaf_value > self.max_Q else self.max_Q
            self.min_Q = leaf_value if leaf_value < self.min_Q else self.min_Q

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(leaf_value)
        leaf_value = float(leaf_value.max(dim=-1)[0].max(dim=-1)[0])
        self.update(leaf_value)

    def get_value(
        self,
        c_puct,
        mean_Q,
        node_value_term,
        node_value_scale,
        prob_term,
        weight_fac=50,
    ):
        # compute value for selection, higher is better
        # check for different variants of prob term
        # alpha zero variant of puct
        self._u = self._calc_prob_term(c_puct, prob_term)
        # orig formula
        # self._u = (c_puct * self._P * math.sqrt(self._parent._n_visits + 1) / (1 + self._n_visits))

        node_value = self._calc_node_value(
            mean_Q, node_value_term, node_value_scale, weight_fac
        )
        value = node_value + self._u
        # print(value)
        return value

    def _calc_prob_term(self, c_puct, prob_term):
        if prob_term == "puct":
            return (
                c_puct
                * self._P
                * math.sqrt(self._parent._n_visits)
                / (1 + self._n_visits)
            )
        elif prob_term == "pucb":
            # two terms
            term_1 = math.sqrt(
                3 * math.log(self._parent._n_visits) / (2 * (self._n_visits + 1))
            )
            term_2 = (2 / self._P) * math.sqrt(
                math.log(self._parent._n_visits) / self._parent._n_visits
            )
            return term_1 - term_2

    def _rescale(self, value, cur_scale, target_scale):
        # assumes value is scaled to [0,1]
        value = (value - cur_scale[0]) / (cur_scale[1] - cur_scale[0])
        value = value * (target_scale[1] - target_scale[0]) + target_scale[0]
        return value

    def _calc_node_value(
        self, mean_Q, node_value_term, node_value_scale, weight_fac=50
    ):
        # calculates just the node value --> includes different variants
        max_Q = self.max_Q
        min_Q = self.min_Q
        parent_max_Q = self._parent.max_Q
        parent_min_Q = self._parent.min_Q
        parent_init_Q = self._parent.q_init
        # should be one if not yet approximated
        if node_value_term == "game":
            # +1 if win -1 if lose against baseline
            scaler = max(parent_max_Q - parent_init_Q, parent_init_Q - parent_min_Q)
            if self._n_visits == 0:
                q = 1
            elif scaler == 0:
                # here the current node must have been visited at least once but yielded the same return as the parent node
                q = 0
            else:
                q = (self._Q - parent_init_Q) / scaler
            q = self._rescale(q, cur_scale=[-1, 1], target_scale=node_value_scale)
        elif node_value_term == "game_asym":
            if self._n_visits == 0:
                q = 1
            elif self._Q == parent_init_Q:
                # here the current node must have been visited at least once but yielded the same return as the parent node
                q = 0
            elif self._Q - parent_init_Q > 0:
                q = (self._Q - parent_init_Q) / (parent_max_Q - parent_init_Q)
            else:
                q = (self._Q - parent_init_Q) / (parent_init_Q - parent_min_Q)
            q = self._rescale(q, cur_scale=[-1, 1], target_scale=node_value_scale)
        elif node_value_term == "game_jump":
            if self._n_visits == 0:
                q = 1
            elif self._Q - parent_init_Q == 0:
                q = 0.0
            elif self._Q > parent_init_Q:
                q = 1
            else:
                q = -1
            q = self._rescale(q, cur_scale=[-1, 1], target_scale=node_value_scale)
        elif node_value_term == "game_clipped":
            if self._n_visits == 0:
                q = 1
            elif self._Q == parent_init_Q:
                q = 0.0
            else:
                # factor 10 corresponds to 10 percent error or improvement clipping per node
                q = np.clip(weight_fac * (1 - self._Q / parent_init_Q), -1, 1)
            q = self._rescale(q, cur_scale=[-1, 1], target_scale=node_value_scale)
        elif node_value_term == "smooth":
            if self._n_visits == 0:
                q = 1
            elif parent_max_Q - parent_min_Q == 0:
                q = 0.5
            else:
                q = 1 - (parent_max_Q - self._Q) / (parent_max_Q - parent_min_Q)
            q = self._rescale(q, cur_scale=[0, 1], target_scale=node_value_scale)
        else:
            # default case
            if self._n_visits == 0:
                q = 1
            # check for different variants
            if parent_max_Q - parent_min_Q == 0:
                # assign zero if the parent node (and potentially sub nodes) have been explored,
                # but giving a return not worse or better than the leaf calculated after first selection
                q = 0
            else:
                # q = -(self._Q - mean_value) / (max_value - min_value)
                # q = np.clip((self._Q - mean_value) / (max_value - min_value), 0, 1)
                q = (self._Q - mean_Q) / (parent_max_Q - parent_min_Q)
                # q = 1 - (max_value - self._Q) / (max_value - min_value) # in [0,1]
            q = self._rescale(q, cur_scale=[-1, 1], target_scale=node_value_scale)
        return q

    def add_virtual_loss(self, virtual_loss):
        if self._parent:
            self._parent.add_virtual_loss(virtual_loss)
        self.n_vlosses += 1
        self._n_visits += virtual_loss

    def revert_virtual_loss(self, virtual_loss):
        if self._parent:
            self._parent.revert_virtual_loss(virtual_loss)
        self.n_vlosses -= 1
        self._n_visits -= virtual_loss

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None

    def __repr__(self) -> str:
        return str(self.__dict__)


class MCTS:
    def __init__(
        self,
        net,
        c_puct=7.5,
        num_playouts=10,
        num_parallel=1,
        virtual_loss=20,
        epsilon=0.91,
        weight_fac=50,
        expansion_limit=None,
        node_value_scale="[-1,1]",
        node_value_term="",
        prob_term="puct",
        aggregation_strategy=0,
    ):
        self._net = net
        self.q_init = None
        self._root = None
        self.cur_best_return = None
        # hyperparameters
        self.virtual_loss = virtual_loss
        self._n_playout = num_playouts
        self.n_parallel = num_parallel
        self._c_puct = c_puct
        self.epsilon = epsilon
        self.weight_fac = weight_fac
        self.expansion_limit = expansion_limit
        # convert string representation of node value scale to list
        self.node_value_scale = [
            int(val) for val in node_value_scale.strip("][").split(",")
        ]
        self.node_value_term = node_value_term
        self.prob_term = prob_term
        self.aggregation_strategy = aggregation_strategy

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
            act, current = current.select(
                self._c_puct,
                self.node_value_term,
                self.node_value_scale,
                self.prob_term,
                self.weight_fac,
            )
            leaf_state = self.env.next_state(leaf_state, act)
            # print(act)
            # print(current.state.selected_node_list)
            # print(leaf_state.selected_node_list)
        return current, leaf_state

    def _playout(self, num_parallel):
        leaves = []
        failsafe = 0
        leaf_states = []
        # select a number of parallel leaves (each can have different number of taken action / depth in the tree)
        while len(leaves) < num_parallel and failsafe < num_parallel:
            failsafe += 1
            leaf, leaf_state = self.select_leaf()
            # add probability of the current leave to exploration rating
            self.exploration_rating += leaf.orig_prob
            if self.env.is_done_state(leaf_state):
                leaf_value = self.env.get_return(leaf_state)
                # increase the number of visits by one for all nodes belonging to this path
                # and update the value along the way
                leaf.update_recursive(leaf_value)
            else:
                leaf.add_virtual_loss(self.virtual_loss)
                leaves.append(leaf)
                leaf_states.append(leaf_state)
            # leaf.add_visits(visits=1)
            # when running the first overall selection break after first iteration
            if self.step == 0 and self.cur_playout == 0:
                break

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
            # Calc priors and values together
            values, priors = self.evaluate_leaves(leaf_states)
            #
            for idx, (leaf, leaf_state, ps, value) in enumerate(
                zip(leaves, leaf_states, priors, values)
            ):
                leaf.revert_virtual_loss(self.virtual_loss)
                ### update_value
                # in 1d case convert tensor to float
                leaf.update_recursive(value)
                ### expand node considering all possible actions
                available_actions = leaf_state.available_actions
                num_ava_actions = available_actions.size(-1)
                # gather the probabilities for all the available actions
                # shape (batch_s, group_s, num_ava_actions)
                prior = torch.gather(ps, dim=-1, index=available_actions)
                prior, indices = torch.sort(prior, dim=-1, descending=True)
                available_actions = torch.gather(
                    available_actions, dim=-1, index=indices
                )
                # expand all the leaves (each leave will be fully expanded)
                leaf.expand(
                    available_actions,
                    prior,
                    self.epsilon,
                    self.aggregation_strategy,
                    self.expansion_limit,
                )
            # check if any new tour is better than current best tour --> fully add nodes to mct

    def evaluate_leaves(self, leaf_states):
        # conversion to multi state
        multi_state = MultiGroupState([leaf_state.copy() for leaf_state in leaf_states])
        # potentially prepare net
        self.prepare_net(len(leaf_states))
        values, priors = self._value_func(multi_state)
        # restore net encoded nodes
        self._net.encoded_nodes = self._net.encoded_nodes[
            0 : multi_state.single_batch_s, :, :
        ]
        return values, priors
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

    def get_move_values(self):
        # add number of current simulations to n_playout to handle sub tree roots which have already been visited
        self.cur_playout = 0
        # print(f"Starting playout number {self.cur_playout}")
        while self.cur_playout < self._n_playout:
            self._playout(self.n_parallel)
            self.cur_playout += 1
        act_values_states = [
            (act, node._Q, node.state, node.orig_prob, node._n_visits)
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
