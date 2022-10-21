from main_code.agents.base_agent import BaseAgent
from main_code.nets.pomo import PomoNetwork
from main_code.utils.utils import AverageMeter
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from main_code.utils.torch_objects import Tensor, LongTensor, device


class PolicyAgent(BaseAgent):
    def __init__(self, policy_net) -> None:
        super().__init__(policy_net)

    def get_action_probabilities(self, state):
        return self.model.get_action_probabilities(state)

    def get_action(self, state):
        action_probs = self.get_action_probabilities(state)
        # shape = (batch, group, TSP_SIZE)
        action = action_probs.argmax(dim=2)
        return action, action_probs

    def learn(self, reward, group_prob_list):
        group_reward = reward
        group_log_prob = group_prob_list.log().sum(dim=2)
        # shape = (batch, group)

        group_advantage = group_reward - group_reward.mean(dim=1, keepdim=True)

        group_loss = -group_advantage * group_log_prob
        # shape = (batch, group)
        loss = group_loss.mean()
        # maybe store optimizer in network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        max_reward, _ = group_reward.max(dim=1)
        return max_reward, group_loss
