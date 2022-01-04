from main_code.agents.base_agent import BaseAgent
from main_code.nets.pomo import PomoNetwork

class PolicyAgent(BaseAgent):
    def __init__(self, policy_net) -> None:
        super().__init__()
        self.model = policy_net
    
    def eval(self):
        self.model.eval()

    def reset(self, state):
        self.model.reset(state)

    def get_action(self, state):
        self.model.update(state)
        action_probs = self.model.get_action_probabilities()
        # shape = (batch, group, TSP_SIZE)
        action = action_probs.argmax(dim=2)
        return action
