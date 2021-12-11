
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nets.encoder.gat_encoder import Encoder, multi_head_attention, reshape_by_heads

class ACTOR(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.box_select_probabilities = None
        # shape = (batch, group, TSP_SIZE)
        # relevant hyperparameters
        self.EMBEDDING_DIM = config.EMBEDDING_DIM

        self.encoder = Encoder(config)
        self.node_prob_calculator = Next_Node_Probability_Calculator_for_group(config)

        self.batch_s = None
        self.encoded_nodes = None

    def reset(self, group_state):
        self.batch_s = group_state.data.size(0)
        self.encoded_nodes = self.encoder(group_state.data)
        # shape = (batch_s, TSP_SIZE, EMBEDDING_DIM)

        self.node_prob_calculator.reset(self.encoded_nodes, group_ninf_mask=group_state.ninf_mask)

    def soft_reset(self, group_state):
        self.node_prob_calculator.reset(self.encoded_nodes, group_ninf_mask=group_state.ninf_mask)

    def update(self, group_state):
        encoded_LAST_NODES = pick_nodes_for_each_group(self.encoded_nodes, 
                                                       group_state.current_node, 
                                                       self.EMBEDDING_DIM)
        # shape = (batch_s, group, EMBEDDING_DIM)

        probs = self.node_prob_calculator(encoded_LAST_NODES)
        # shape = (batch_s, group, TSP_SIZE)
        self.box_select_probabilities = probs

    def get_action_probabilities(self):
        return self.box_select_probabilities



########################################
# ACTOR_SUB_NN : Next_Node_Probability_Calculator
########################################

class Next_Node_Probability_Calculator_for_group(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.HEAD_NUM = config.HEAD_NUM
        self.KEY_DIM = config.KEY_DIM
        self.EMBEDDING_DIM = config.EMBEDDING_DIM
        self.TSP_SIZE = config.TSP_SIZE
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

    def reset(self, encoded_nodes, group_ninf_mask):
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

    def forward(self, encoded_LAST_NODE):
        # encoded_LAST_NODE.shape = (batch_s, group, EMBEDDING_DIM)

        if self.q_first is None:
            self.q_first = reshape_by_heads(self.Wq_first(encoded_LAST_NODE), head_num=self.HEAD_NUM)
        # shape = (batch_s, HEAD_NUM, group, KEY_DIM)

        #  Multi-Head Attention
        #######################################################
        q_last = reshape_by_heads(self.Wq_last(encoded_LAST_NODE), head_num=self.HEAD_NUM)
        # shape = (batch_s, HEAD_NUM, group, KEY_DIM)

        q = self.q_graph + self.q_first + q_last
        # shape = (batch_s, HEAD_NUM, group, KEY_DIM)

        out_concat = multi_head_attention(q, self.k, self.v, group_ninf_mask=self.group_ninf_mask, tsp_size=self.TSP_SIZE)
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

        score_masked = score_clipped + self.group_ninf_mask.clone()

        probs = F.softmax(score_masked, dim=2)
        # shape = (batch_s, group, TSP_SIZE)

        return probs

def pick_nodes_for_each_group(encoded_nodes, node_index_to_pick, embed_dim):
    # encoded_nodes.shape = (batch_s, TSP_SIZE, EMBEDDING_DIM)
    # node_index_to_pick.shape = (batch_s, group_s)
    batch_s = node_index_to_pick.size(0)
    group_s = node_index_to_pick.size(1)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_s, group_s, embed_dim)
    # shape = (batch_s, group, EMBEDDING_DIM)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape = (batch_s, group, EMBEDDING_DIM)

    return picked_nodes