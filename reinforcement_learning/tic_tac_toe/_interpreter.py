from .._templates import Interpreter as _Interpreter
import torch


class Interpreter(_Interpreter):

    @staticmethod
    def repr(states):
        """
        draw the given states
        """
        icons = [" X ", "   ", " O "]
        columns = [" A ", " B ", " C "]
        return "\n\n".join(("\n"+"  "+"-"*11+"\n").join(f"{3 - i} " + "|".join(icons[i+1] for i in row) for i, row in enumerate(state)) + "\n" + "  " + " ".join(col for col in columns) for state in states)

    @staticmethod
    def initial_state() -> torch.Tensor:
        return torch.zeros((1, 3, 3), dtype=torch.long)

    @staticmethod
    def rewards(states: torch.Tensor, actions: torch.Tensor, new_states: torch.Tensor) -> torch.Tensor:
        """
        rewards the player for it's actions

        Parameters
        ----------
        environment : Environment
            state before action
        action : Action
            action of the agent
        new_environment : Environment
            state after action

        Returns
        -------
        torch.Tensor :
            tensor of floats of shape (N)
        """
        return (Interpreter.game_is_over(new_states) & ~Interpreter.game_is_over(states)).float()

    @staticmethod
    def apply(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Returns a copy of the environment after the actions have been applied
        """
        device = states.device
        y, x = actions[:, 0].to(device), actions[:, 1].to(device)
        new_states = states.detach().clone()
        new_states[torch.arange(len(new_states), device=device), y, x] = 1
        return new_states

    @staticmethod
    def change_turn(states: torch.Tensor) -> torch.Tensor:
        """
        change the turn of the state
        """
        return -states

    @staticmethod
    def valid_plays_mask(states: torch.Tensor) -> torch.Tensor:
        """
        returns a boolean mask to apply to Q values to obtain valid plays

        Parameters
        ----------
        states : torch.Tensor
            tensor of shape (N, Ly, Lx) of Q values for each action
        """
        return states == 0

    @staticmethod
    def action_is_valid(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        returns True if the given action is valid
        """
        mask = Interpreter.valid_plays_mask(state)
        y, x = action[:, 0], action[:, 1]
        return mask[:, y, x]

    @staticmethod
    def game_is_over(states: torch.Tensor) -> torch.Tensor:
        """
        return a tensor of booleans defining if the game is over
        """
        N = states.shape[0]
        all_filled = (states != 0).reshape(N, -1).all(dim=1)
        return Interpreter._current_player_won(states) | Interpreter._other_player_won(states) | all_filled

    @staticmethod
    def _current_player_won(states: torch.Tensor) -> torch.Tensor:
        """
        return a tensor of booleans defining if the game is over
        """
        player_pawns = (states == 1)
        return player_pawns.all(dim=2).any(dim=1) | player_pawns.all(dim=1).any(dim=1) | player_pawns[:, [0, 1, 2], [0, 1, 2]].all(dim=1) | player_pawns[:, [0, 1, 2], [2, 1, 0]].all(dim=1)

    @staticmethod
    def _other_player_won(states: torch.Tensor) -> torch.Tensor:
        """
        return a tensor of booleans defining if the game is over
        """
        player_pawns = (states == -1)
        return player_pawns.all(dim=2).any(dim=1) | player_pawns.all(dim=1).any(dim=1) | player_pawns[:, [0, 1, 2], [0, 1, 2]].all(dim=1) | player_pawns[:, [0, 1, 2], [2, 1, 0]].all(dim=1)
