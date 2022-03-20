from .._templates import Action as _Action
import torch


class Action(_Action):

    columns = ["A", "B", "C"]

    @classmethod
    def from_string(cls, string: str) -> "Action":
        x, y = string.upper()
        return Action(torch.tensor([cls.columns.index(x)]),
                      torch.tensor([int(y)-1]))
    
    def __repr__(self):
        return "\n".join(f"{self.columns[x]}{3-y}" for x, y in zip(self.x, self.y))

    def __init__(self, y: torch.Tensor, x: torch.Tensor):
        self.y = y
        self.x = x