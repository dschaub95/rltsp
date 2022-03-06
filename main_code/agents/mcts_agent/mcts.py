import numpy as np
import torch
import math
from main_code.agents.mcts_agent.tree_node import TreeNode
from main_code.environment.environment_new import (
    MultiGroupState,
    GroupState,
    GroupEnvironment,
)


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
