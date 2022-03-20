from ._environment import Environment
from ._action import Action
import torch

class Interpreter:

    @classmethod
    def rewards(cls, environment: Environment, action: Action, new_environment: Environment) -> torch.Tensor:
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
        raise NotImplementedError()
