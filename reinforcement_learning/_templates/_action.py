
class Action:
    
    @classmethod
    def from_string(cls, string: str) -> "Action":
        raise NotImplementedError()

    def __init__(self):
        pass