import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from main_code.nets.utils.multi_head_attention import reshape_by_heads, multi_head_attention


class MHADecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.HEAD_NUM = config.HEAD_NUM
        self.KEY_DIM = config.KEY_DIM
        self.EMBEDDING_DIM = config.EMBEDDING_DIM
        self.LOGIT_CLIPPING = config.LOGIT_CLIPPING

        self.Wq_graph = nn.Linear(self.EMBEDDING_DIM, self.HEAD_NUM * self.KEY_DIM, bias=False)
        self.Wq_first = nn.Linear(self.EMBEDDING_DIM, self.HEAD_NUM * self.KEY_DIM, bias=False)
        self.Wq_last = nn.Linear(self.EMBEDDING_DIM, self.HEAD_NUM * self.KEY_DIM, bias=False)
        self.Wk = nn.Linear(self.EMBEDDING_DIM, self.HEAD_NUM * self.KEY_DIM, bias=False)
        self.Wv = nn.Linear(self.EMBEDDING_DIM, self.HEAD_NUM * self.KEY_DIM, bias=False)

        self.multi_head_combine = nn.Linear(self.HEAD_NUM * self.KEY_DIM, self.EMBEDDING_DIM)

        self.q_graph = None  # saved q1, for multi-head attention
        self.q_first = None  # saved q2, for multi-head attention
        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.group_ninf_mask = None  # reference to ninf_mask owned by state

    def reset(self, encoded_nodes, group_ninf_mask=None):
        # this function saves some time by making some operations only once after a new graph was encoded
        # encoded_nodes.shape = (batch_s, TSP_SIZE, EMBEDDING_DIM)

        encoded_graph = encoded_nodes.mean(dim=1, keepdim=True)
        # shape = (batch_s, 1, EMBEDDING_DIM)
        self.q_graph = reshape_by_heads(self.Wq_graph(encoded_graph), head_num=self.HEAD_NUM)
        # shape = (batch_s, HEAD_NUM, 1, KEY_DIM)
        self.q_first = None
        # shape = (batch_s, HEAD_NUM, group, KEY_DIM)
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=self.HEAD_NUM)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=self.HEAD_NUM)
        # shape = (batch_s, HEAD_NUM, TSP_SIZE, KEY_DIM)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape = (batch_s, EMBEDDING_DIM, TSP_SIZE)
        self.group_ninf_mask = group_ninf_mask
        # shape = (batch_s, group, TSP_SIZE)

    def forward(self, encoded_first_node, encoded_last_node, group_ninf_mask):
        # encoded_last_node.shape = (batch_s, group, EMBEDDING_DIM)

        if self.q_first is None:
            self.q_first = reshape_by_heads(self.Wq_first(encoded_first_node), head_num=self.HEAD_NUM)
        # shape = (batch_s, HEAD_NUM, group, KEY_DIM)

        #  Multi-Head Attention
        #######################################################
        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=self.HEAD_NUM)
        # shape = (batch_s, HEAD_NUM, group, KEY_DIM)

        q = self.q_graph + self.q_first + q_last
        # shape = (batch_s, HEAD_NUM, group, KEY_DIM)

        out_concat = multi_head_attention(q, self.k, self.v, group_ninf_mask=group_ninf_mask)
        # shape = (batch_s, group, HEAD_NUM*KEY_DIM)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape = (batch_s, group, EMBEDDING_DIM)

        #  Single-Head Attention, for probability calculation
        #######################################################      
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape = (batch_s, group, TSP_SIZE)

        score_scaled = score / np.sqrt(self.EMBEDDING_DIM)
        # shape = (batch_s, group, TSP_SIZE)

        score_clipped = self.LOGIT_CLIPPING * torch.tanh(score_scaled)
        
        score_masked = score_clipped + group_ninf_mask.clone()

        probs = F.softmax(score_masked, dim=2)
        # shape = (batch_s, group, TSP_SIZE)

        return probs


# different heads for value, policy and Q-value network