from .._templates import Action as _Action
import torch


class Action(_Action):

    columns = ["A", "B", "C"]

    @classmethod
    def from_string(cls, string: str) -> "Action":
        x, y = string.upper()
        return Action(torch.tensor([cls.columns.index(x)]),
                      torch.tensor([3 - int(y)]))
    
    def __repr__(self):
        return "\n".join(f"{self.columns[x]}{3 - y}" for x, y in zip(self.x, self.y))

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y