class BaseAgent:
    def __init__(self, net) -> None:
        self.model = net

    def eval(self):
        self.model.eval()

    def reset(self, state):
        self.model.reset(state)

    def fine_tune(self, data_loader):
        raise NotImplementedError

    def get_action(self, state):
        raise NotImplementedError
