from .._templates import Environement as _Environement
import torch

class Environment(_Environement):

    def __init__(self):
        super().__init__()

    @classmethod
    def initial_state(cls) -> torch.Tensor:
        """
        Returns the initial state of a new game
        """
        return torch.zeros((1, 3, 3), dtype=torch.long)

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
        return torch.masked_fill(Q_values, states < 0, -float("inf"))
    
    @classmethod
    def next_states(cls, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        returns the next states given actions of the agent

        Parameters
        ----------
        states : torch.Tensor
            tensor of shape (N, Ly, Lx) of board states
        actions : torch.Tensor
            tensor of shape (N, 2) of agent actions
        """
        copy = states.detach().clone()
        copy[:, actions[:, 0], actions[:, 1]] = 1
        return copy

    @classmethod
    def game_is_over(cls, states: torch.Tensor) -> torch.Tensor:
        """
        return a tensor of booleans defining if the game is over
        """
        player = (states == 1)
        return player.all(dim=2).any(dim=1) | player.all(dim=1).any(dim=1) | player[:, [0, 1, 2], [0, 1, 2]].all(dim=1) | player[:, [0, 1, 2], [2, 1, 0]].all(dim=1)

    @classmethod
    def draw(cls, state: torch.Tensor):
        """
        draw the given state
        """
        icons = [" X ", "   ", " O "]
        return ("\n"+"-"*11+"\n").join("|".join(icons[i+1] for i in row) for row in state)
