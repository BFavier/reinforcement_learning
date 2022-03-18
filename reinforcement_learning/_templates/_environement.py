import torch
from typing import Optional

class Environement:
    """
    The environement is what the agent can interact with
    """

    def __init__(self):
        self.states: torch.Tensor = self.initial_state()

    def apply(self, actions: torch.Tensor):
        self.next_states(self.states)
    
    @classmethod
    def initial_state(cls) -> torch.Tensor:
        """
        Returns the initial state of a new game
        """
        raise NotImplementedError()

    @classmethod
    def mask_invalid_plays(cls, states: torch.Tensor, Q_values: torch.Tensor) -> torch.Tensor:
        """
        masks the invalid plays with a Q value of -infinity

        Parameters
        ----------
        states : torch.Tensor
            tensor of shape (N, Ly, Lx) of board state
        Q_value : torch.Tensor
            tensor of shape (N, *, Ly, Lx) of Q values for each action
        """
        raise NotImplementedError()
    
    @classmethod
    def next_states(cls, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        returns the next states given actions of the agent

        Parameters
        ----------
        states : torch.Tensor
            tensor of shape (N, *, Ly, Lx) of board states
        actions : torch.Tensor
            tensor of shape (N, *, 2) of agent actions
        """
        raise NotImplementedError()

    @classmethod
    def game_is_over(cls, states: torch.Tensor) -> torch.Tensor:
        """
        return a tensor of booleans defining if the game is over
        """
        raise NotImplementedError()

    @classmethod
    def draw(cls, state: torch.Tensor):
        """
        draw the given state

        Parameters
        ----------
        state : torch.Tensor
            tensor of shape (Ly, Lx)
        """
        raise NotImplementedError()