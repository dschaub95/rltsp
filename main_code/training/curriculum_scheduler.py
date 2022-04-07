import math
import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


class CurriculumScheduler:
    def __init__(self, start, stop, step_epoch, step_size=1) -> None:
        assert start <= stop
        self.start = start
        self.stop = stop
        self.step_epoch = step_epoch
        self.step_size = step_size

    def __call__(self, epoch):
        tot_steps = max(math.floor(epoch - 1 / self.step_epoch), 0)
        return min(self.start + tot_steps * self.step_size, self.stop)


class StochasticCurriculumScheduler:
    def __init__(self, start, stop, stddev) -> None:
        self.start = start
        self.stop = stop
        self.stddev = stddev
        self.default_vector = np.arange(start=start, stop=stop + 1)
        self.rng = np.random.default_rng(7337)

    def __call__(self, epoch):
        # calculate gaussian based on default vector
        result_vector = (1 / (np.sqrt(2 * np.pi) * self.stddev)) * np.exp(
            -(1 / 2) * ((self.default_vector - epoch) / self.stddev) ** 2
        )
        # apply softmax to result vector
        result_probs = softmax(result_vector)
        # sample index from result vector based on probabilities
        sampled_size = self.rng.choice(self.default_vector, 1, p=result_probs)[0]
        # return size
        return sampled_size
