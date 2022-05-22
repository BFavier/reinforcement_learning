import torch

class Interpreter:

    @staticmethod
    def rewards(states: torch.Tensor, actions: torch.Tensor, new_states: torch.Tensor) -> torch.Tensor:
        """
        rewards the player for it's actions
        """
        raise NotImplementedError()

    @staticmethod
    def initial_state() -> torch.Tensor:
        """
        generates an initial state to play from
        """
        raise NotImplementedError()

    @staticmethod
    def apply(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Returns a copy of the environment after the actions have been applied
        """
        raise NotImplementedError()

    @staticmethod
    def change_turn(states: torch.Tensor) -> torch.Tensor:
        """
        change the turn of the state
        """
        raise NotImplementedError()

    @staticmethod
    def valid_plays_mask(states: torch.Tensor) -> torch.Tensor:
        """
        returns a boolean mask to apply to Q values to obtain valid plays
        """
        raise NotImplementedError()

    @staticmethod
    def action_is_valid(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        returns True if the given action is valid
        """
        raise NotImplementedError()

    @staticmethod
    def game_is_over(states: torch.Tensor) -> torch.Tensor:
        """
        return a tensor of booleans defining if the game is over
        """
        raise NotImplementedError()
