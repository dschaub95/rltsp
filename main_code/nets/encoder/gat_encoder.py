import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from main_code.nets.utils.multi_head_attention import reshape_by_heads, multi_head_attention

# taken from Joshi et al.

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Linear(2, config.EMBEDDING_DIM)
        self.layers = nn.ModuleList([Encoder_Layer(config) for _ in range(config.ENCODER_LAYER_NUM)])

    def forward(self, data):
        # data.shape = (batch_s, TSP_SIZE, 2)

        embedded_input = self.embedding(data)
        # shape = (batch_s, TSP_SIZE, EMBEDDING_DIM)

        out = embedded_input
        for layer in self.layers:
            out = layer(out)

        return out


class Encoder_Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.HEAD_NUM = config.HEAD_NUM
        self.KEY_DIM = config.KEY_DIM
        self.EMBEDDING_DIM = config.EMBEDDING_DIM
        self.TSP_SIZE = config.TSP_SIZE
        self.FF_HIDDEN_DIM = config.FF_HIDDEN_DIM

        self.Wq = nn.Linear(self.EMBEDDING_DIM, self.HEAD_NUM * self.KEY_DIM, bias=False)
        self.Wk = nn.Linear(self.EMBEDDING_DIM, self.HEAD_NUM * self.KEY_DIM, bias=False)
        self.Wv = nn.Linear(self.EMBEDDING_DIM, self.HEAD_NUM * self.KEY_DIM, bias=False)
        self.multi_head_combine = nn.Linear(self.HEAD_NUM * self.KEY_DIM, self.EMBEDDING_DIM)

        self.addAndNormalization1 = Add_And_Normalization_Module(config)
        self.feedForward = Feed_Forward_Module(self.EMBEDDING_DIM, self.FF_HIDDEN_DIM)
        self.addAndNormalization2 = Add_And_Normalization_Module(config)

    def forward(self, input1):
        # input.shape = (batch_s, TSP_SIZE, EMBEDDING_DIM)

        q = reshape_by_heads(self.Wq(input1), head_num=self.HEAD_NUM)
        k = reshape_by_heads(self.Wk(input1), head_num=self.HEAD_NUM)
        v = reshape_by_heads(self.Wv(input1), head_num=self.HEAD_NUM)
        # q shape = (batch_s, HEAD_NUM, TSP_SIZE, KEY_DIM)

        out_concat = multi_head_attention(q, k, v, self.TSP_SIZE)
        # shape = (batch_s, TSP_SIZE, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape = (batch_s, TSP_SIZE, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3

class Add_And_Normalization_Module(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.EMBEDDING_DIM = config.EMBEDDING_DIM
        self.TSP_SIZE = config.TSP_SIZE
        self.norm_by_EMB = nn.BatchNorm1d(self.EMBEDDING_DIM, affine=True)
        # 'Funny' Batch_Norm, as it will normalized by EMB dim

    def forward(self, input1, input2):
        # input.shape = (batch_s, TSP_SIZE, EMBEDDING_DIM)

        batch_s = input1.size(0)
        added = input1 + input2
        normalized = self.norm_by_EMB(added.reshape(batch_s * self.TSP_SIZE, self.EMBEDDING_DIM))

        return normalized.reshape(batch_s, self.TSP_SIZE, self.EMBEDDING_DIM)


class Feed_Forward_Module(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(embed_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, input1):
        # input.shape = (batch_s, TSP_SIZE, EMBEDDING_DIM)

        return self.W2(F.relu(self.W1(input1)))