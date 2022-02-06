from main_code.agents.base_agent import BaseAgent
from main_code.nets.pomo import PomoNetwork


class PolicyAgent(BaseAgent):
    def __init__(self, policy_net) -> None:
        super().__init__(policy_net)

    def get_action_probabilities(self, state):
        return self.model.get_action_probabilities(state)

    def get_action(self, state):
        action_probs = self.get_action_probabilities(state)
        # shape = (batch, group, TSP_SIZE)
        action = action_probs.argmax(dim=2)
        return action
