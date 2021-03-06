from main_code.nets.pomo import PomoNetwork


class BaseAgent:
    def __init__(self, net) -> None:
        self.model: PomoNetwork = net

    def eval(self):
        self.model.eval()

    def reset(self, state):
        # encode a new state
        self.model.reset(state)

    def learn(self):
        raise NotImplementedError

    def get_action(self, state):
        raise NotImplementedError
