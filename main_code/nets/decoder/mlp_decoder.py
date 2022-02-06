import torch
import torch.nn.functional as F
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, hidden_dim, output_dim, L=2):
        super(MLP, self).__init__()
        self.L = L
        U = []
        for layer in range(self.L - 1):
            U.append(nn.Linear(hidden_dim, hidden_dim, True))
        self.U = nn.ModuleList(U)
        self.V = nn.Linear(hidden_dim, output_dim, True)

    def forward(self, x):
        Ux = x
        for U_i in self.U:
            Ux = U_i(Ux)
            Ux = F.relu(Ux)
        y = self.V(Ux)
        return y
