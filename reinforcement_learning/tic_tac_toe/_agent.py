import torch

from .._templates import Agent as _Agent
from ._interpreter import Interpreter


class Agent(_Agent):

    interpreter = Interpreter

    def __init__(self, dim: int=16, activation: str="relu", gamma: float=0.9):
        super().__init__(gamma=gamma)
        self.dim = dim
        self.embedding = torch.nn.Embedding(3, dim)
        self.convolution = torch.nn.Conv2d(dim, dim, (3, 3))
        self.activation = getattr(torch, activation)
        self.deconvolution = torch.nn.ConvTranspose2d(dim, 1, (3, 3))

    def q(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Returns the expected cumulated rewards of the given action
        """
        Q = self.Q(states)
        device = Q.device
        y, x = actions[:, 0], actions[:, 1]
        q = Q[:, y.to(device), x.to(device)]
        q = torch.masked_fill(q, self.interpreter.game_is_over(states).to(device), 0.)
        return q

    def Q(self, states: torch.Tensor):
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

    def play(self, states: torch.Tensor, epsilon: float = 0.) -> torch.Tensor:
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
        Q = self.Q(states)
        valid_plays = self.interpreter.valid_plays_mask(states).to(Q.device)
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
        return torch.stack([indices_y, indices_x], dim=1)

