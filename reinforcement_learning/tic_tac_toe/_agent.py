import torch

from typing import Iterable
from .._templates import Agent as _Agent
from ._interpreter import Interpreter
from itertools import combinations, chain


class Agent(_Agent):

    interpreter = Interpreter

    def __init__(self, dim: int=16, activation: str="relu", gamma: float=0.9):
        super().__init__(gamma=gamma)
        self.dim = dim
        self.embedding = torch.nn.Embedding(3, dim)
        self.contract = torch.nn.Linear(9*dim, dim)
        self.activation = getattr(torch, activation)
        self.expand = torch.nn.Linear(dim, 9)
    
    def _apply_transforms(self, X: torch.Tensor) -> Iterable[torch.Tensor]:
        transforms = [lambda x: torch.flip(x, [-2]),
                      lambda x: torch.flip(x, [-3]),
                      lambda x: torch.transpose(x, -2, -3)]
        for trans in chain(*[combinations(transforms, k) for k in range(len(transforms)+1)]):
            Y = X
            for tran in trans:
                Y = tran(Y)
            yield Y

    def q(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Returns the expected cumulated rewards of the given action
        """
        Q = self.Q(states)
        device = Q.device
        y, x = actions[:, 0], actions[:, 1]
        q = Q[torch.arange(len(actions), device=Q.device), y.to(device), x.to(device)]
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
        X = self.embedding(X + 1)
        X = torch.stack(list(self._apply_transforms(X)), dim=0)
        R, N, Ly, Lx, D = X.shape
        X = X.reshape(R, N, -1)
        X = self.contract(X)
        X = self.activation(X)
        X = self.expand(X).reshape(R, N, 3, 3)
        return torch.tanh(X).max(dim=0).values

    def play(self, states: torch.Tensor, epsilon: float = 0.) -> torch.Tensor:
        """
        Given an environment returns the next environment state

        Parameters
        ----------
        states : torch.Tensor
            the initial state
        epsilon : float
            the probability of making a random choice (between 0. and 1.)

        Returns
        -------
        tuple of torch.Tensor:
            (action, q) the action of the agent and the expected cumulated reward
        """
        Q = self.Q(states)
        valid_plays = self.interpreter.valid_plays_mask(states).to(Q.device)
        Q = torch.masked_fill(Q, ~valid_plays, -float("inf"))
        N, Ly, Lx = Q.shape
        q, indices = Q.reshape(N, -1).max(dim=1)
        indices_y = indices // Lx
        indices_x = indices % Lx
        # if there needs to be some random play with probability epsilon
        if epsilon > 0.:
            # get indexes of best plays according to a random Q mapping
            random_Q = torch.rand(Q.shape, device=Q.device)
            random_Q = torch.masked_fill(random_Q, ~valid_plays, -float("inf"))
            rand_q, rand_indices = random_Q.reshape(N, -1).max(dim=1)
            random_y = rand_indices // Lx
            random_x = rand_indices % Lx
            # generate a random number to remplace or not by the random action
            replace = (torch.rand(random_x.shape, device=random_x.device) < epsilon) & torch.isfinite(rand_q)
            indices_y[replace] = random_y[replace]
            indices_x[replace] = random_x[replace]
            q[replace] = rand_q
        action = torch.stack([indices_y, indices_x], dim=1)
        return action, q

