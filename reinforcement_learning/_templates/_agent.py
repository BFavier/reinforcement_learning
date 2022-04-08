import torch
from itertools import combinations
from typing import List, Callable, Tuple
from ._environment import Environment
from ._action import Action

class Agent(torch.nn.Module):

    transformations = None

    def __init__(self, gamma: float = 0.9):
        if not 0. < gamma < 1.0:
            raise ValueError("The 'gamma' parameter must be between 0. and 1. excluded")
        self.gamma = gamma
        if not hasattr(self.transformations, "__iter__"):
            raise NotImplementedError("Expected the attribute 'transformations' "
                                      "to be defined as an iterable of transform functions")
        self.symetries = [list(transforms) for k in range(len(self.transformations)+1)
                          for transforms in combinations(self.transformations, k)]
        super().__init__()
    
    def copy(self) -> "Agent":
        """
        create a copy of the agent
        """
        raise NotImplementedError()
    
    def q(self, environment: Environment, action: Action) -> torch.Tensor:
        """
        Returns the expected cumulated rewards of the given action
        """
        raise NotImplementedError()

    def play(self, environment: Environment, epsilon: float = 0.) -> Action:
        """
        Given an environment returns the next environment state

        Parameters
        ----------
        environment : Environment
            the initial environment
        epsilon : float
            the probability of making a suboptimal choice (between 0. and 1.)

        Returns
        -------
        Action:
            the action of the agent
        """
        raise NotImplementedError()

    def _Q_function(self, states: torch.Tensor) -> torch.Tensor:
        """
        The Q function gives the predicted maximum cumulated reward for each action

        Parameters
        ----------
        states : torch.Tensor
            tensor of longs of shape (N, Ly, Lx)
        
        Returns
        -------
        torch.Tensor :
            tensor of floats of shape (N, *, Ly, Lx)
        """
        raise NotImplementedError()

    def Q(self, environment: Environment) -> torch.Tensor:
        """
        Apply the problem's symetries to the Q function and get the maximum
        Q value for each action
        """
        symetries = [self._Q_with_transforms(environment.states, transforms) for transforms in self.symetries]
        return torch.max(torch.stack(symetries, -1), dim=-1).values

    def _Q_with_transforms(self, states: torch.Tensor, transforms: List[Callable]) -> torch.Tensor:
        """
        Apply the given transforms before and after computing the Q function
        """
        for t in transforms:
            states = t(states)
        X = self._Q_function(states)
        for t in transforms[::-1]:
            X = t(X)
        return X
