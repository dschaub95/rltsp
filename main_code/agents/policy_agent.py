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
        agent_info = None
        return action, agent_info

    def learn(self, reward, group_prob_list):
        group_reward = reward
        group_log_prob = group_prob_list.log().sum(dim=2)
        # shape = (batch, group)

        group_advantage = group_reward - group_reward.mean(dim=1, keepdim=True)

        group_loss = -group_advantage * group_log_prob
        # shape = (batch, group)
        loss = group_loss.mean()

        self.policy_net.optimizer.zero_grad()
        loss.backward()
        self.policy_net.optimizer.step()
        max_reward, _ = group_reward.max(dim=1)
        return max_reward, group_loss

    def fine_tune(
        self,
        data_loader,
        env_model,
        tsp_size,
        group_s,
        num_epochs=8,
        lr_rate=1e-4,
        weight_decay=1e-6,
        lr_decay_epoch=1.0,
        lr_decay_gamma=1.0,
    ):
        # at best take a dataloader based on the test dataset with the correct batchsize
        # otherwise init custom dataloader from a batch of data with the test batch size
        # (is strongly dependent on the test batch size and thus more complicated)
        # freeze encoder weights
        # enable training on decoder weights
        self.policy_net.optimizer = optim.Adam(
            self.policy_net.parameters(),  # restrict parameters
            lr=lr_rate,
            weight_decay=weight_decay,
        )
        self.policy_net.lr_stepper = lr_scheduler.StepLR(
            self.policy_net.optimizer, step_size=lr_decay_epoch, gamma=lr_decay_gamma
        )
        distance_AM = AverageMeter()
        actor_loss_AM = AverageMeter()
        episode = 0
        for epoch in range(num_epochs):
            for batch in data_loader:
                # calculate node embeddings once --> can be made more efficient by calculating this for an independent batch size
                # node embeddings should only be calculated in the first epoch and then saved

                # implement policy gradient training algorithm

                batch_s = batch.size(0)
                episode = episode + batch_s

                # Actor Group Move
                ###############################################
                env = env_model(batch, tsp_size)
                group_state, reward, done = env.reset(group_size=group_s)
                self.policy_net.reset(group_state)

                # First Move is given
                first_action = LongTensor(np.arange(group_s))[None, :].expand(
                    batch_s, group_s
                )
                group_state, reward, done = env.step(first_action)

                group_prob_list = Tensor(np.zeros((batch_s, group_s, 0)))
                while not done:
                    # actor_group.update(group_state)
                    action_probs = self.policy_net.get_action_probabilities(group_state)
                    # shape = (batch, group, TSP_SIZE)
                    action = (
                        action_probs.reshape(batch_s * group_s, -1)
                        .multinomial(1)
                        .squeeze(dim=1)
                        .reshape(batch_s, group_s)
                    )
                    # shape = (batch, group)
                    group_state, reward, done = env.step(action)

                    batch_idx_mat = torch.arange(batch_s)[:, None].expand(
                        batch_s, group_s
                    )
                    group_idx_mat = torch.arange(group_s)[None, :].expand(
                        batch_s, group_s
                    )
                    chosen_action_prob = action_probs[
                        batch_idx_mat, group_idx_mat, action
                    ].reshape(batch_s, group_s)
                    # shape = (batch, group)
                    group_prob_list = torch.cat(
                        (group_prob_list, chosen_action_prob[:, :, None]), dim=2
                    )
                max_reward, group_loss = self.learn(reward, group_prob_list)
                distance_AM.push(-max_reward)  # reward was given as negative dist
                actor_loss_AM.push(group_loss.detach().reshape(-1))
                # compare with opt len (for tuning on validation set)
                # log metrics with wandb
                pass
        pass
