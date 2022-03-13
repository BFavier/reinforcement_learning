import torch

from ..common.agent import Agent

class TicTacToePlayer(Agent):

    def __init__(self, dim: int = 16, activation: str = "relu"):
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
        N = states.shape[0]
        X = self.embedding(states).permute(0, 2, 3, 1)
        X = self.convolution(X)
        X = self.activation(X)
        X = self.deconvolution(X).unsqueeze(1)
        return X

