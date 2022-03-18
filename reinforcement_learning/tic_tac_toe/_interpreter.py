from .._templates import Interpreter as _Interpreter
from ._environement import Environment
import torch


class Interpreter(_Interpreter):

    def __init__(self):
        super().__init__()

    def rewards(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        """
        rewards the player for it's actions

        Parameters
        ----------
        states : torch.Tensor
            tensor of shape (N, Ly, Lx)
        actions : torch.Tensor
            tensor of shapes (N, *, 2)
        next_states : torch.Tensor
            tensor of shape (N, Ly, Lx)

        Returns
        -------
        torch.Tensor :
            tensor of floats of shape (N)
        """
        return (Environment.game_is_over(next_states) & ~Environment.game_is_over(states)).float()

        