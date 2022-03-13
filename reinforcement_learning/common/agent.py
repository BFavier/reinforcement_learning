import torch
from itertools import combinations
from typing import List, Callable

class Agent(torch.nn.Module):

    def __inti__(self):
        super().__init__()

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.Q_function_with_symetries(states)

    def Q_function(self, states: torch.Tensor):
        """
        computes the Q function of each action for each given state

        Parameters
        ----------
        state : torch.Tensor
            tensor of longs of shape (N, 3, 3)
        
        Returns
        -------
        torch.Tensor :
            tensor of floats of shape (N, 3, 3)
        """
        raise NotImplementedError()

    def Q_function_with_symetries(self, states: torch.Tensor) -> torch.Tensor:
        """
        Apply the problem's symetries to the Q function

        Parameters
        ----------
        """
        transformations = self._all_combinations(self._transposed, self._x_flipped, self._y_flipped)
        symetries = [self._Q_with_transforms(states, transforms) for transforms in transformations]
        return torch.max(torch.stack(symetries, -1), dim=-1)

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
    
    def _all_combinations(*args):
        return [list(transforms) for k in range(len(args)+1)
                for transforms in combinations(*args, k)]

    @staticmethod
    def _transposed(self, states: torch.Tensor) -> torch.Tensor:
        return states.transpose(-1, -2)
    
    @staticmethod
    def _x_flipped(self, states: torch.Tensor) -> torch.Tensor:
        return states.flip(-1)

    @staticmethod
    def _y_flipped(self, states: torch.Tensor) -> torch.Tensor:
        return states.flip(-2)