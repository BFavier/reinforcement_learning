import torch
import copy
from typing import Tuple

from .._templates import Agent as _Agent
from ._environment import Environment
from ._action import Action


class Agent(_Agent):

    def __init__(self, dim: int = 16, activation: str = "relu"):
        self.transformations = [self._transposed, self._x_flipped, self._y_flipped]
        super().__init__()
        self.dim = dim
        self.embedding = torch.nn.Embedding(3, dim)
        self.convolution = torch.nn.Conv2d(dim, dim, (3, 3))
        self.activation = getattr(torch, activation)
        self.deconvolution = torch.nn.ConvTranspose2d(dim, 1, (3, 3))

    def copy(self) -> "Agent":
        """
        returns a copy of the agent
        """
        agent = Agent(self.dim)
        agent.to(self.convolution.weight.device)
        agent.activation = self.activation
        agent.load_state_dict(copy.deepcopy(self.state_dict()))
        return agent
    
    def _Q_function(self, states: torch.Tensor):
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
        device = self.embedding.weight.device
        X = states.to(device)
        X = self.embedding(X + 1).permute(0, 3, 1, 2)
        X = self.convolution(X)
        X = self.activation(X)
        X = self.deconvolution(X).squeeze(1)
        return X

    def _choose_action(self, environment: Environment, Q_values: torch.Tensor) -> Tuple[Action, torch.Tensor]:
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
        tuple :
            the tuple (action, q) with q the expected cumulated sum of rewards
        """
        N, Ly, Lx = Q_values.shape
        q, indices = Q_values.reshape(N, -1).max(dim=1)
        indices_y = indices // Lx
        indices_x = indices % Lx
        action = Action(indices_y, indices_x)
        return action, q
    
    @staticmethod
    def _transposed(states: torch.Tensor) -> torch.Tensor:
        return states.transpose(-1, -2)
    
    @staticmethod
    def _x_flipped(states: torch.Tensor) -> torch.Tensor:
        return states.flip(-1)

    @staticmethod
    def _y_flipped(states: torch.Tensor) -> torch.Tensor:
        return states.flip(-2)

