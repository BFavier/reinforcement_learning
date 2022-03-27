from .._templates import Environment as _Environment
from ._action import Action
import torch
from typing import Optional
import numpy as np
import math

class Environment(_Environment):

    def __repr__(self):
        """
        draw the given state
        """
        icons = [" X ", "   ", " O "]
        columns = [" A ", " B ", " C "]
        return "\n\n".join(("\n"+"  "+"-"*11+"\n").join(f"{3 - i} " + "|".join(icons[i+1] for i in row) for i, row in enumerate(state)) + "\n" + "  " + " ".join(col for col in columns) for state in self.states)

    def __init__(self, states: Optional[torch.Tensor] = None):
        super().__init__()
        if states is None:
            self.states = torch.zeros((1, 3, 3), dtype=torch.long)
        else:
            self.states = states

    def __getitem__(self, indexing) -> "Environment":
        """
        index the given set of environments
        """
        return Environment(states=self.states[indexing])

    def apply(self, action: Action) -> "Environment":
        """
        Returns a copy of the environment after the actions have been applied
        """
        new_states = self.states.detach().clone()
        new_states[torch.arange(len(new_states), device=action.y.device), action.y, action.x] = 1
        return Environment(states=new_states)

    def valid_plays_mask(self) -> torch.Tensor:
        """
        returns a boolean mask to apply to Q values to obtain valid plays

        Parameters
        ----------
        Q_value : torch.Tensor
            tensor of shape (N, *, Ly, Lx) of Q values for each action
        """
        return self.states == 0

    def action_is_valid(self, action: Action) -> torch.Tensor:
        """
        returns True if the given action is valid
        """
        mask = self.valid_plays_mask()
        y, x = action.yx[:, 0], action.yx[:, 1]
        return mask[:, y, x]
    
    def current_player_won(self) -> torch.Tensor:
        """
        return a tensor of booleans defining if the game is over
        """
        player_pawns = (self.states == 1)
        return player_pawns.all(dim=2).any(dim=1) | player_pawns.all(dim=1).any(dim=1) | player_pawns[:, [0, 1, 2], [0, 1, 2]].all(dim=1) | player_pawns[:, [0, 1, 2], [2, 1, 0]].all(dim=1)

    def other_player_won(self) -> torch.Tensor:
        """
        return a tensor of booleans defining if the game is over
        """
        player_pawns = (self.states == -1)
        return player_pawns.all(dim=2).any(dim=1) | player_pawns.all(dim=1).any(dim=1) | player_pawns[:, [0, 1, 2], [0, 1, 2]].all(dim=1) | player_pawns[:, [0, 1, 2], [2, 1, 0]].all(dim=1)

    def game_is_over(self) -> torch.Tensor:
        """
        return a tensor of booleans defining if the game is over
        """
        N = self.states.shape[0]
        all_filled = (self.states != 0).reshape(N, -1).all(dim=1)
        return self.current_player_won() | self.other_player_won() | all_filled

    def extend(self, environment: "Environment") -> "Environment":
        """
        extend this environment set with another environment set
        """
        return Environment(states=torch.cat([self.states, environment.states], dim=0))

    def sample(self, n: int) -> "Environment":
        """
        returns a sample of 'n' or less oservations of the given environment set
        """
        # n = min(n, len(self.states))
        # indexes = torch.randperm(len(self.states))
        # return self[indexes[:n]]

        n = min(n, len(self.states))
        N = len(self.states)
        # bucket observations by number of plays in it
        buckets = {L: self.states[(self.states != 0).reshape(N, -1).sum(dim=1) == L] for L in range(9)}
        # hash in each bucket similar plays
        buckets = {k: self._filter_uniques(v) for k, v in buckets.items()}
        # sample from each bucket
        samples = []
        for i in range(9):
            subset = buckets[i]
            k = min(len(subset), math.ceil(n/(9 - i)))
            indexes = torch.randperm(len(subset))
            samples.append(subset[indexes[:k]])
            n -= k
        return Environment(states=torch.cat(samples, dim=0))

    def _hash(self, states: torch.Tensor) -> torch.Tensor:
        """
        returns a hash of a given board state
        """
        N = len(states)
        if N == 0:
            return states
        mask = 3**torch.arange(9).reshape(1, 3, 3)
        return ((states+1) * mask).reshape(N, -1).sum(dim=1)

    def _filter_uniques(self, states: torch.Tensor) -> torch.Tensor:
        """
        """
        array = self._hash(states).detach().numpy()
        _, index = np.unique(array, return_index=True)
        index = torch.from_numpy(index)
        return states[index]
