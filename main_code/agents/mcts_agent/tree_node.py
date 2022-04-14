import math
import numpy as np


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
