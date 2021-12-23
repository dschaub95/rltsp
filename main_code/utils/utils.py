import numpy as np
import torch
from main_code.utils.torch_objects import device


class Average_Meter:
    """
    Class to keep track of the length of the current tour
    """
    def __init__(self):
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.sum = torch.tensor(0.).to(device)
        self.count = 0

    def push(self, some_tensor, n_for_rank_0_tensor=None):
        assert not some_tensor.requires_grad # You get Memory error, if you keep tensors with grad history
        
        rank = len(some_tensor.shape)

        if rank == 0: # assuming "already averaged" Tensor was pushed
            self.sum += some_tensor * n_for_rank_0_tensor
            self.count += n_for_rank_0_tensor
            
        else:
            self.sum += some_tensor.sum()
            self.count += some_tensor.numel()

    def peek(self):
        average = (self.sum / self.count).tolist()
        return average

    def result(self):
        average = (self.sum / self.count).tolist()
        self.reset()
        return average

