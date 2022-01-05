
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from main_code.nets.encoder.gat_encoder import Encoder
from main_code.nets.decoder.mha_decoder import MHADecoder


class PomoNetwork(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.box_select_probabilities = None
        # shape = (batch, group, TSP_SIZE)
        # relevant hyperparameters
        self.EMBEDDING_DIM = config.EMBEDDING_DIM

        self.encoder = Encoder(config)
        self.node_prob_calculator = MHADecoder(config)

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