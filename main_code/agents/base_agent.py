class BaseAgent:
    def __init__(self, net) -> None:
        self.model = net

    def eval(self):
        self.model.eval()

    def reset(self, state):
        self.model.reset(state)

    def get_action(self, state):
        raise NotImplementedError
