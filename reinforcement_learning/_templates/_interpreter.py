import torch

class Interpreter:

    def __init__(self):
        pass

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
        raise NotImplementedError()
