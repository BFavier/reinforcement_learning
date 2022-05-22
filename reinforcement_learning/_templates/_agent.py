import torch
from ._interpreter import Interpreter

class Agent(torch.nn.Module):

    interpreter = Interpreter

    def __init__(self, gamma: float=0.9):
        super().__init__()
        self.gamma = gamma

    def q(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Returns the expected cumulated rewards of the given action
        """
        raise NotImplementedError()

    def Q(self, states: torch.Tensor):
        """
        computes the Q function of each action for each given state
        """
        raise NotImplementedError()

    def play(self, states: torch.Tensor, epsilon: float = 0.) -> torch.Tensor:
        """
        Given an environment returns the next environment state
        """
        raise NotImplementedError()
