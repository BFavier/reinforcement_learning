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
        create a copy of the agent
        """
        obj = Agent(self.dim, self.activation.__name__)
        obj.load_state_dict(self.state_dict())
        obj.to(self.convolution.weight.device)
        return obj
    
    def q(self, environment: Environment, action: Action) -> torch.Tensor:
        """
        Returns the expected cumulated rewards of the given action
        """
        Q = self.Q(environment)
        return Q[:, action.y, action.x]

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
        Q = self.Q(environment)
        valid_plays = environment.valid_plays_mask().to(Q.device)
        Q = torch.masked_fill(Q, ~valid_plays, -float("inf"))
        N, Ly, Lx = Q.shape
        indices = Q.reshape(N, -1).max(dim=1).indices
        indices_y = indices // Lx
        indices_x = indices % Lx
        # get indexes of best plays according to a random Q mapping
        random_Q = torch.rand(Q.shape, device=Q.device)
        random_Q = torch.masked_fill(random_Q, ~valid_plays, -1.)
        rand_q, rand_indices = random_Q.reshape(N, -1).max(dim=1)
        random_y = rand_indices // Lx
        random_x = rand_indices % Lx
        # generate a random number to remplace or not by the random action
        replace = (torch.rand(random_x.shape, device=random_x.device) < epsilon) & (rand_q >= 0.)
        indices_y[replace] = random_y[replace]
        indices_x[replace] = random_x[replace]
        return Action(indices_x, indices_y)

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
        return torch.tanh(X)

    @staticmethod
    def _transposed(states: torch.Tensor) -> torch.Tensor:
        return states.transpose(-1, -2)
    
    @staticmethod
    def _x_flipped(states: torch.Tensor) -> torch.Tensor:
        return states.flip(-1)

    @staticmethod
    def _y_flipped(states: torch.Tensor) -> torch.Tensor:
        return states.flip(-2)

