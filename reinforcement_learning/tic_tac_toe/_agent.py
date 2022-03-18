import torch
from typing import Tuple

from .._templates import Agent as _Agent


class Agent(_Agent):

    def __init__(self, dim: int = 16, activation: str = "relu"):
        self.transformations = [self._transposed, self._x_flipped, self._y_flipped]
        super().__init__()
        self.dim = dim
        self.embedding = torch.nn.Embedding(3, dim)
        self.convolution = torch.nn.Conv2d(dim, dim, (3, 3))
        self.activation = getattr(torch, activation)
        self.deconvolution = torch.nn.ConvTranspose2d(dim, 1, (3, 3))
    
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
        X = self.embedding(states + 1).permute(0, 3, 1, 2)
        X = self.convolution(X)
        X = self.activation(X)
        X = self.deconvolution(X).squeeze(1)
        return X
    
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
            actions of shape (N, 2)
        """
        N, Ly, Lx = Q_values.shape
        indices = Q_values.reshape(N, -1).max(dim=1).indices
        indices_y = indices // Lx
        indices_x = indices % Lx
        actions = torch.stack((indices_y, indices_x), dim=1)
        return actions
    
    @staticmethod
    def _transposed(states: torch.Tensor) -> torch.Tensor:
        return states.transpose(-1, -2)
    
    @staticmethod
    def _x_flipped(states: torch.Tensor) -> torch.Tensor:
        return states.flip(-1)

    @staticmethod
    def _y_flipped(states: torch.Tensor) -> torch.Tensor:
        return states.flip(-2)

