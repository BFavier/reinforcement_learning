import torch
from itertools import combinations
from typing import List, Callable, Tuple

class Agent(torch.nn.Module):

    transformations = None

    def __init__(self):
        if not hasattr(self.transformations, "__iter__"):
            raise NotImplementedError("Expected the attribute 'transformations' "
                                      "to be defined as an iterable of transform functions")
        self.symetries = [list(transforms) for k in range(len(self.transformations)+1)
                          for transforms in combinations(self.transformations, k)]
        super().__init__()
    
    def play(self, states: torch.Tensor, Q_values: torch.Tensor) -> torch.Tensor:
        """
        Given a state and the computed Q_values, returns the actions and next state

        Parameters
        ----------
        states : torch.Tensor
            tensor of longs of shape (N, Ly, Lx)
        Q_values : torch.Tensor
            tensor of floats of shape (N, *, Ly, Lx)
        
        Returns
        -------
        torch.Tensor :
            actions of shape (N, *, 2)
        """
        raise NotImplementedError()

    def Q_function(self, states: torch.Tensor) -> torch.Tensor:
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

    def Q_with_symetries(self, states: torch.Tensor) -> torch.Tensor:
        """
        Apply the problem's symetries to the Q function and get the maximum
        Q value for each action
        """
        symetries = [self._Q_with_transforms(states, transforms) for transforms in self.symetries]
        return torch.max(torch.stack(symetries, -1), dim=-1).values

    def _Q_with_transforms(self, states: torch.Tensor, transforms: List[Callable]) -> torch.Tensor:
        """
        Apply the given transforms before and after computing the Q function
        """
        for t in transforms:
            states = t(states)
        X = self.Q_function(states)
        for t in transforms[::-1]:
            X = t(X)
        return X

