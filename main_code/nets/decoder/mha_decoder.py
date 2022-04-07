import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from main_code.nets.utils.multi_head_attention import (
    reshape_by_heads,
    multi_head_attention,
)


class MHADecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_num = config.HEAD_NUM
        self.key_dim = config.KEY_DIM
        self.embedding_dim = config.EMBEDDING_DIM
        self.logit_clipping = config.LOGIT_CLIPPING
        # TODO needs refactoring
        try:
            self.embed_graph = config.embed_graph
        except:
            self.embed_graph = True

        if self.embed_graph:
            self.Wq_graph = nn.Linear(
                self.embedding_dim, self.head_num * self.key_dim, bias=False
            )
        self.Wq_first = nn.Linear(
            self.embedding_dim, self.head_num * self.key_dim, bias=False
        )
        self.Wq_last = nn.Linear(
            self.embedding_dim, self.head_num * self.key_dim, bias=False
        )
        self.Wk = nn.Linear(
            self.embedding_dim, self.head_num * self.key_dim, bias=False
        )
        self.Wv = nn.Linear(
            self.embedding_dim, self.head_num * self.key_dim, bias=False
        )

        self.multi_head_combine = nn.Linear(
            self.head_num * self.key_dim, self.embedding_dim
        )

        self.q_graph = None  # saved q1, for multi-head attention
        self.q_first = None  # saved q2, for multi-head attention
        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.group_ninf_mask = None  # reference to ninf_mask owned by state

    def reset(self, encoded_nodes, group_ninf_mask=None):
        # this function saves some time by making some operations only once after a new graph was encoded
        # encoded_nodes.shape = (batch_s, TSP_SIZE, embedding_dim)

        encoded_graph = encoded_nodes.mean(dim=1, keepdim=True)
        # shape = (batch_s, 1, embedding_dim)
        if self.embed_graph:
            self.q_graph = reshape_by_heads(
                self.Wq_graph(encoded_graph), head_num=self.head_num
            )
        # shape = (batch_s, head_num, 1, key_dim)
        self.q_first = None
        # shape = (batch_s, head_num, group, key_dim)
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=self.head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=self.head_num)
        # shape = (batch_s, head_num, TSP_SIZE, key_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape = (batch_s, embedding_dim, TSP_SIZE)
        self.group_ninf_mask = group_ninf_mask
        # shape = (batch_s, group, TSP_SIZE)

    def forward(self, encoded_first_node, encoded_last_node, group_ninf_mask):
        # encoded_last_node.shape = (batch_s, group, embedding_dim)

        if self.q_first is None:
            self.q_first = reshape_by_heads(
                self.Wq_first(encoded_first_node), head_num=self.head_num
            )
        # shape = (batch_s, head_num, group, key_dim)

        #  Multi-Head Attention
        #######################################################
        q_last = reshape_by_heads(
            self.Wq_last(encoded_last_node), head_num=self.head_num
        )
        # shape = (batch_s, head_num, group, key_dim)
        if self.embed_graph:
            q = self.q_graph + self.q_first + q_last
        else:
            q = self.q_first + q_last
        # shape = (batch_s, head_num, group, key_dim)

        out_concat = multi_head_attention(
            q, self.k, self.v, group_ninf_mask=group_ninf_mask
        )
        # shape = (batch_s, group, head_num*key_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape = (batch_s, group, embedding_dim)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape = (batch_s, group, TSP_SIZE)

        score_scaled = score / np.sqrt(self.embedding_dim)
        # shape = (batch_s, group, TSP_SIZE)

        score_clipped = self.logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + group_ninf_mask.clone()

        probs = F.softmax(score_masked, dim=2)
        # shape = (batch_s, group, TSP_SIZE)

        return probs


# different heads for value, policy and Q-value network
