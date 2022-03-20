import torch
from typing import Optional, Sequence
from ._action import Action

class Environment:
    """
    The environment is what the agent can interact with
    """

    def __init__(self):
        self.states: Optional[torch.Tensor] = None

    def __getitem__(self, indexing) -> "Environment":
        """
        index the given set of environments
        """
        raise NotImplementedError()

    def __len__(self):
        """
        returns the length of the environment set
        """
        return len(self.states)

    def apply(self, action: Action) -> "Environment":
        """
        Returns a copy of the environment after the actions have been applied
        """
        raise NotImplementedError()

    def change_turn(self) -> "Environment":
        """
        for turn-by-turn two players game, return environment with negated
        states
        """
        return type(self)(-self.states)

    def valid_plays_mask(self, Q_values: torch.Tensor) -> torch.Tensor:
        """
        returns a boolean mask to apply to Q values to obtain valid plays

        Parameters
        ----------
        Q_value : torch.Tensor
            tensor of shape (N, *, Ly, Lx) of Q values for each action
        """
        raise NotImplementedError()

    def action_is_valid(self, action: Action) -> torch.Tensor:
        """
        returns True if the given action is valid
        """
        raise NotImplementedError()

    def game_is_over(self) -> torch.Tensor:
        """
        return a tensor of booleans defining if the game is over
        """
        raise NotImplementedError()
    
    def extend(self, environment: "Environment"):
        """
        extend this environment set with another environment set
        """
        raise NotImplementedError()
    
    def sample(self, n: int) -> "Environment":
        """
        returns a sample of 'n' or less oservations of the given environment set
        """
        raise NotImplementedError()
    
    def sample(self, n: int) -> "Environment":
        """
        returns a sample of 'n' or less oservations of the given environment set
        """
        n = min(n, len(self.states))
        indexes = torch.randperm(len(self.states))
        return self[indexes[:n]]